"""Offline IBQ tokens for LIBERO's deterministic two-camera observations."""
import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


def file_stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def cache_recipe(cfg):
    u0 = cfg.framework.u0
    if cfg.datasets.vla_data.video_backend != "torchvision_av":
        raise ValueError("U0 cache requires the verified torchvision_av frame contract")
    if int(u0.image_size) <= 0 or int(u0.image_size) % 16:
        raise ValueError("U0 cache image_size must be a positive multiple of 16")
    files = [Path(u0.vision_tokenizer) / name for name in ("config.yaml", "model.ckpt")]
    recipe = {"version": 3, "representation": "ibq_codebook_grid", "codebook_size": 131072, "image_size": int(u0.image_size), "precision": "fp32",
              "preprocess": "PIL RGB resize224 bicubic then resize-image_size bicubic",
              "files": [file_stamp(path) for path in files]}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:20]


def source_stamp(ds, episode):
    paths = [ds.get_video_path(episode, key.removeprefix("video.")) for key in ds.modality_keys["video"]]
    paths += [ds.dataset_path / ds.data_path_pattern.format(
        episode_chunk=ds.get_episode_chunk(episode), episode_index=episode)]
    paths += [ds.dataset_path / "meta" / name for name in ("info.json", "modality.json")]
    return [file_stamp(path) for path in paths]


def cache_path(root, recipe, name):
    return Path(root) / recipe / name


def validate_codes(codes, length, views, grid_shape=None):
    if (codes.ndim != 4 or codes.shape[:2] != (length, views)
            or codes.dtype != np.int32 or (grid_shape is not None and codes.shape[2:] != tuple(grid_shape))
            or not all(codes.shape[2:]) or np.any(codes < 0) or np.any(codes >= 131072)):
        raise ValueError(f"Invalid cached vision codes: {codes.shape}, {codes.dtype}")


class CachedLiberoDataset(LeRobotSingleDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        recipe = self.data_cfg["u0_vision_cache_recipe"]
        self.cache_dir = cache_path(self.data_cfg["u0_vision_cache_dir"], recipe, self.dataset_name)
        manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        if manifest["recipe"] != recipe or manifest["views"] != self.modality_keys["video"]:
            raise ValueError("Incompatible U0 vision cache")
        for episode, length in zip(self.trajectory_ids, self.trajectory_lengths):
            entry = manifest["episodes"][str(int(episode))]
            if entry["length"] != int(length) or entry["sources"] != source_stamp(self, int(episode)):
                raise ValueError(f"Stale U0 cache for episode {episode}; regenerate cache")
            if not (self.cache_dir / f"episode_{episode:06d}.npy").is_file():
                raise FileNotFoundError(f"Missing U0 cache episode {episode}")
        self._cached_episode = None
        self._cached_tokens = None
        self._grid_shape = manifest["grid_shape"]

    def get_training_sample(self, trajectory_id, base_index):
        if self._cached_episode != trajectory_id:
            tokens = np.load(self.cache_dir / f"episode_{trajectory_id:06d}.npy", mmap_mode="r", allow_pickle=False)
            length = int(self.trajectory_lengths[self.get_trajectory_index(trajectory_id)])
            validate_codes(tokens, length, len(self.modality_keys["video"]), self._grid_shape)
            self._cached_episode, self._cached_tokens = trajectory_id, tokens
        data = self.transforms(self.get_step_data(trajectory_id, base_index, skip_video=True))
        return self._pack_sample(data, image_codes=self._cached_tokens[base_index].tolist())


def dataset_specs(cfg):
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
    mix = cfg.datasets.vla_data.data_mix
    if mix not in ("libero_goal", "libero_all"):
        raise ValueError("Offline cache supports libero_goal and libero_all")
    specs = DATASET_NAMED_MIXTURES[mix]
    if any(robot != "libero_franka" for _, _, robot in specs):
        raise ValueError("U0 cache requires the LIBERO Franka image contract")
    return specs


def make_dataset(cfg, dataset_name=None):
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.datasets.vla_data, resolve=True))
    data_cfg.pop("u0_vision_cache_dir", None)
    specs = dataset_specs(cfg)
    if dataset_name is None and len(specs) == 1:
        dataset_name = specs[0][0]
    for name, _, robot in specs:
        if name == dataset_name:
            return make_LeRobotSingleDataset(Path(data_cfg.data_root_dir), name, robot,
                                            delete_pause_frame=data_cfg.get("delete_pause_frame", False), data_cfg=data_cfg)
    raise ValueError(f"Select a dataset from the mixture: {dataset_name!r}")


def worker(rank, cfg_dict, workers, batch_size, limit, dataset_name):
    import torch
    import torchvision
    from PIL import Image
    from starVLA.model.modules.vlm.Emu3_5 import Emu3_5Interface
    from starVLA.model.modules.vlm.emu3_5.vision_tokenizer import build_vision_tokenizer

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    cfg = OmegaConf.create(cfg_dict)
    ds = make_dataset(cfg, dataset_name)
    recipe = cache_recipe(cfg)
    out = cache_path(cfg.datasets.vla_data.u0_vision_cache_dir, recipe, ds.dataset_name)
    out.mkdir(parents=True, exist_ok=True)
    u0 = cfg.framework.u0
    from types import SimpleNamespace
    encoder = SimpleNamespace(image_size=int(u0.image_size))
    encoder.vq_model = build_vision_tokenizer("ibq", u0.vision_tokenizer, device=f"cuda:{rank}").requires_grad_(False).eval()
    torchvision.set_video_backend("pyav")
    episodes = list(zip(ds.trajectory_ids, ds.trajectory_lengths))
    if limit:
        episodes = episodes[:limit]
    for episode, length in episodes[rank::workers]:
        episode, length = int(episode), int(length)
        target = out / f"episode_{episode:06d}.npy"
        stamp = source_stamp(ds, episode)
        meta = target.with_suffix(".json")
        if target.exists() and meta.exists() and json.loads(meta.read_text()) == stamp:
            validate_codes(np.load(target, mmap_mode="r", allow_pickle=False), length, 2)
            print(f"[cache rank {rank}] episode {episode}: reuse", flush=True)
            continue
        timestamps = ds.get_trajectory_data(episode)["timestamp"].to_numpy()
        views = []
        for key in ds.modality_keys["video"]:
            path = ds.get_video_path(episode, key.removeprefix("video."))
            reader = torchvision.io.VideoReader(str(path), "video")
            frames, pts = [], []
            for frame in reader:
                frames.append(frame["data"].permute(1, 2, 0).numpy())
                pts.append(frame["pts"])
            del reader
            if not frames:
                raise ValueError(f"Empty video: {path}")
            indices = np.abs(np.asarray(pts)[:, None] - timestamps[None, :]).argmin(axis=0)
            images = [Image.fromarray(frames[i]).resize((224, 224)) for i in indices]
            # Check sequential decode against the existing seek-based training path.
            ds.curr_traj_data = ds.get_trajectory_data(episode)
            for step in {0, length // 2, length - 1}:
                online = Image.fromarray(ds.get_video(episode, key, step)[0]).resize((224, 224))
                if not np.array_equal(np.asarray(online), np.asarray(images[step])):
                    raise ValueError(f"Offline/online frame mismatch: {episode}, {key}, {step}")
            # Reuse the exact online preprocessing and tensor layout. Even a
            # different batch-one stride can select different FP32 kernels/IDs.
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                encoded = [Emu3_5Interface.encode_image_codes(encoder, image).cpu().numpy() for image in images]
            views.append(encoded)
        tokens = np.asarray(views, dtype=np.int32).transpose(1, 0, 2, 3)
        validate_codes(tokens, length, 2)
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as file:
            np.save(file, tokens, allow_pickle=False)
        os.replace(temporary, target)
        meta_temp = meta.with_suffix(f".{os.getpid()}.tmp")
        meta_temp.write_text(json.dumps(stamp))
        os.replace(meta_temp, meta)
        print(f"[cache rank {rank}] episode {episode}: {length} frames, {tokens.shape}", flush=True)


@lru_cache(maxsize=1)
def _legacy_layout(tokenizer, grid_shape):
    import torch
    from starVLA.model.modules.vlm.Emu3_5 import _format_image_tokens
    visual_ids = np.asarray([tokenizer.convert_tokens_to_ids(f"<|visual token {i:06d}|>")
                             for i in range(131072)], dtype=np.int64)
    if len(np.unique(visual_ids)) != len(visual_ids) or np.any(visual_ids < 0):
        raise ValueError("Tokenizer does not provide a unique IBQ visual vocabulary")
    inverse = np.full(max(len(tokenizer), int(visual_ids.max()) + 1), -1, dtype=np.int32)
    inverse[visual_ids] = np.arange(131072, dtype=np.int32)
    template = np.asarray(tokenizer.encode(_format_image_tokens(tokenizer, torch.zeros(grid_shape, dtype=torch.long)),
                                         add_special_tokens=False))
    visual = inverse[template] >= 0
    return inverse, template, visual


def extract_codebook_grids(formatted, tokenizer, grid_shape):
    """Strict inverse of the old image formatting; rejects malformed wrappers."""
    inverse, template, visual = _legacy_layout(tokenizer, tuple(grid_shape))
    if (formatted.ndim != 3 or formatted.dtype != np.int32 or formatted.shape[-1] != len(template)
            or np.any(formatted < 0) or np.any(formatted >= len(inverse))
            or visual.sum() != int(np.prod(grid_shape))
            or not np.all(formatted[..., ~visual] == template[~visual])):
        raise ValueError("Malformed formatted image cache")
    codes = inverse[formatted[..., visual]]
    if np.any(codes < 0):
        raise ValueError("Non-visual token in cached image grid")
    return codes.reshape(*formatted.shape[:2], *grid_shape)


def migrate_formatted_cache(cfg, ds, old, out):
    from types import SimpleNamespace
    from starVLA.model.modules.vlm.Emu3_5 import Emu3_5Interface
    from starVLA.model.modules.vlm.u0.tokenization_unis import UNISTokenizer
    u0 = cfg.framework.u0
    tokenizer_path = Path(u0.get("tokenizer_path", u0.base_vlm))
    # Reconstruct the version-2 identity, including its text tokenizer dependency.
    files = [Path(u0.vision_tokenizer) / name for name in ("config.yaml", "model.ckpt")]
    files += [tokenizer_path / name for name in ("unis.tiktoken", "unis_vision_tokens.txt")]
    legacy = {"version": 2, "image_size": int(u0.image_size), "precision": "fp32",
              "preprocess": "PIL RGB resize224 bicubic then resize-image_size bicubic",
              "files": [file_stamp(path) for path in files]}
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()[:20]
    manifest = json.loads((old / "manifest.json").read_text())
    if manifest["recipe"] != expected or manifest["views"] != ds.modality_keys["video"]:
        raise ValueError("Legacy cache does not match the encoder/tokenizer/preprocessing")
    tokenizer = UNISTokenizer(vocab_file=str(tokenizer_path / "unis.tiktoken"),
                              special_tokens_file=str(tokenizer_path / "unis_vision_tokens.txt"))
    Emu3_5Interface._set_special_tokens(SimpleNamespace(tokenizer=tokenizer))
    grid_shape = (int(u0.image_size) // 16,) * 2
    out.mkdir(parents=True, exist_ok=True)
    for episode, length in zip(ds.trajectory_ids, ds.trajectory_lengths):
        episode, length = int(episode), int(length)
        stamp = source_stamp(ds, episode)
        entry = manifest["episodes"][str(episode)]
        if entry["sources"] != stamp or entry["length"] != length:
            raise ValueError(f"Stale legacy source for episode {episode}")
        source = old / f"episode_{episode:06d}.npy"
        codes = extract_codebook_grids(np.load(source, allow_pickle=False), tokenizer, grid_shape)
        validate_codes(codes, length, 2, grid_shape)
        target = out / source.name
        temp = target.with_suffix(f".{os.getpid()}.tmp")
        with temp.open("wb") as file:
            np.save(file, codes, allow_pickle=False)
        os.replace(temp, target)
        meta = target.with_suffix(".json")
        temp = meta.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(stamp))
        os.replace(temp, meta)
    print(f"Losslessly migrated {len(ds.trajectory_ids)} episodes to raw IBQ grids: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--migrate-formatted-cache", type=Path, help="Convert a verified version-2 formatted cache without re-encoding")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-episodes", type=int, default=0, help="Diagnostic only; does not publish a complete manifest")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch-size must be positive")
    if args.batch_size != 1:
        parser.error("Use batch-size 1: batched FP32 kernels can change discrete IBQ IDs")
    cfg = OmegaConf.load(args.config)
    specs = dataset_specs(cfg)
    if args.migrate_formatted_cache and (len(specs) != 1 or args.limit_episodes):
        parser.error("Migration requires one complete dataset")
    for name, _, _ in specs:
        prepare_dataset_cache(cfg, args, name)


def prepare_dataset_cache(cfg, args, dataset_name):
    ds = make_dataset(cfg, dataset_name)  # Initialize shared statistics before starting workers.
    recipe = cache_recipe(cfg)
    out = cache_path(cfg.datasets.vla_data.u0_vision_cache_dir, recipe, ds.dataset_name)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not args.limit_episodes:
        manifest = json.loads(manifest_path.read_text())
        if manifest["recipe"] != recipe or manifest["views"] != ds.modality_keys["video"]:
            raise ValueError("Incompatible cache manifest")
        valid = True
        for episode, length in zip(ds.trajectory_ids, ds.trajectory_lengths):
            entry = manifest["episodes"].get(str(int(episode)), {})
            path = out / f"episode_{episode:06d}.npy"
            if entry.get("length") != int(length) or entry.get("sources") != source_stamp(ds, int(episode)) or not path.exists():
                valid = False
                break
            validate_codes(np.load(path, mmap_mode="r", allow_pickle=False), int(length), 2, manifest["grid_shape"])
        if valid:
            print(f"Verified existing U0 cache: {out}", flush=True)
            return
    if args.migrate_formatted_cache:
        migrate_formatted_cache(cfg, ds, args.migrate_formatted_cache, out)
    else:
        import torch.multiprocessing as mp
        mp.spawn(worker, args=(OmegaConf.to_container(cfg, resolve=True), args.workers, args.batch_size, args.limit_episodes, dataset_name),
                 nprocs=args.workers, join=True)
    if args.limit_episodes:
        return
    recipe = cache_recipe(cfg)
    out = cache_path(cfg.datasets.vla_data.u0_vision_cache_dir, recipe, ds.dataset_name)
    entries, grid_shape = {}, (int(cfg.framework.u0.image_size) // 16,) * 2
    for episode, length in zip(ds.trajectory_ids, ds.trajectory_lengths):
        path = out / f"episode_{episode:06d}.npy"
        tokens = np.load(path, mmap_mode="r", allow_pickle=False)
        validate_codes(tokens, int(length), 2, grid_shape)
        stamp = source_stamp(ds, int(episode))
        if json.loads(path.with_suffix(".json").read_text()) != stamp:
            raise ValueError(f"Source changed during cache generation: {episode}")
        entries[str(int(episode))] = {"length": int(length), "sources": stamp}
    manifest = {"recipe": recipe, "views": ds.modality_keys["video"], "grid_shape": grid_shape, "representation": "ibq_codebook_grid", "codebook_size": 131072, "episodes": entries}
    temp = out / f"manifest.{os.getpid()}.tmp"
    temp.write_text(json.dumps(manifest))
    os.replace(temp, out / "manifest.json")
    print(f"Complete U0 cache: {out}, {len(entries)} episodes", flush=True)


if __name__ == "__main__":
    main()
