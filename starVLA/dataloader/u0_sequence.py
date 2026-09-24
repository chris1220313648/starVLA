"""LIBERO same-trajectory windows; raw current state, existing action transforms."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.u0_vision_cache import CachedLiberoDataset, dataset_specs, make_dataset, cache_recipe, file_stamp
from starVLA.model.modules.vlm.u0_sequence import contract_hash


class SequenceLiberoDataset(CachedLiberoDataset):
    def sample_window(self, rng):
        if not self.all_steps:
            raise ValueError('No complete action windows in this dataset')
        return self.all_steps[int(rng.integers(len(self.all_steps)))]

    def _get_all_steps(self):
        h = int(self.data_cfg['sequence_h'])
        horizon = int(self.data_cfg.get('sequence_action_horizon', 8))
        if h < 1:
            raise ValueError('sequence_h must be positive')
        return [(int(ep), t) for ep, length in zip(self.trajectory_ids, self.trajectory_lengths)
                for t in range(max(0, int(length) - horizon * h + 1))]

    def _get_delta_indices(self):
        result = super()._get_delta_indices()
        for key in self.modality_keys['state']:
            result[key] = np.array([0])
        return result

    def get_training_sample(self, trajectory_id, base_index):
        h = int(self.data_cfg['sequence_h'])
        horizon = int(self.data_cfg.get('sequence_action_horizon', 8))
        length = int(self.trajectory_lengths[self.get_trajectory_index(trajectory_id)])
        if base_index < 0 or base_index + horizon * h > length:
            raise ValueError('Action window crosses trajectory boundary')
        segments = [super(SequenceLiberoDataset, self).get_training_sample(trajectory_id, base_index + horizon * i) for i in range(h)]
        # Read raw values directly: metadata's pad/gripper names are both joint positions.
        states = np.stack(self.curr_traj_data['observation.state'].to_numpy()).astype(np.float32)
        sample = segments[0]
        sample.update(sequence_action=np.stack([s['action'] for s in segments]),
                      sequence_image_codes=np.stack([s['image_codes'] for s in segments]),
                      sequence_state=states[base_index:base_index + horizon * h:horizon],
                      trajectory_id=int(trajectory_id), frame_index=int(base_index))
        if self.data_cfg.get('oat_tokenizer_path'):
            from starVLA.model.modules.action_model.oat_ActionHeader import OATNormalization
            if not hasattr(self, '_oat_normalization'):
                self._oat_normalization = OATNormalization(self.data_cfg['oat_tokenizer_path'])
            raw = np.stack(self.curr_traj_data['action'].to_numpy()).astype(np.float32)
            actions = self._oat_normalization.normalize(raw[base_index:base_index+h*horizon])
            sample['sequence_action'] = actions.reshape(h, horizon, 7)
            sample['action'] = sample['sequence_action'][0]
        return sample


def prepare(config, output, h):
    cfg = OmegaConf.load(config)
    oat = cfg.framework.name in ('U0OAT', 'u0oat')
    horizon = 32 if oat else 8
    cfg.datasets.vla_data.sequence_action_horizon = horizon
    if oat:
        source = Path(cfg.framework.action_model.oat_tokenizer_path).resolve()
        target = Path(output).resolve().parent / 'oat'
        target.mkdir(parents=True, exist_ok=True)
        for name in ('model.safetensors', 'config.json', 'norm_stats.pt', 'policy_io.json', 'train_config.yaml', 'README.md'):
            if source / name != target / name:
                shutil.copyfile(source / name, target / name)
        cfg.framework.action_model.oat_tokenizer_path = str(target)
        cfg.datasets.vla_data.oat_tokenizer_path = str(target)
    cfg.framework.u0.sequence_h = h
    cfg.datasets.vla_data.sequence_h = h
    cfg.datasets.vla_data.u0_vision_cache_recipe = cache_recipe(cfg)
    arrays, sources, windows = [], [], {}
    for name, _, _ in dataset_specs(cfg):
        ds = make_dataset(cfg, name)
        entries = []
        for ep, length in zip(ds.trajectory_ids, ds.trajectory_lengths):
            frame = ds.get_trajectory_data(int(ep))
            values = np.stack(frame['observation.state'].to_numpy()).astype(np.float32)
            if values.shape != (int(length), 8) or not np.isfinite(values).all():
                raise ValueError(f'Invalid raw state: {name}/{ep}')
            arrays.append(values)
            path = ds.dataset_path / ds.data_path_pattern.format(episode_chunk=ds.get_episode_chunk(int(ep)), episode_index=int(ep))
            sources.append(file_stamp(path))
            entries.append([int(ep), int(length), max(0, int(length) - horizon*h + 1)])
        windows[name] = entries
    states = np.concatenate(arrays)
    contract = dict(version=1, sequence_h=h, action_horizon=horizon, action_dim=7, state_dim=8,
                    state_order=['eef_x', 'eef_y', 'eef_z', 'axis_angle_x', 'axis_angle_y', 'axis_angle_z', 'gripper_left', 'gripper_right'],
                    state_encoding='q01_q99_clip_floor256_constant128',
                    state_statistics={'q01': np.quantile(states, .01, axis=0).tolist(), 'q99': np.quantile(states, .99, axis=0).tolist()},
                    prompt_template='You are a helpful assistant for vla task. USER: Given the current camera views, how to perform the following task? {instruction} State: {state} ',
                    action_normalization='starvla_libero_franka_existing_transforms', action_token_start=149595,
                    image_order=['primary_image', 'wrist_image'], image_rotation=180,
                    image_preprocess='RGB_PIL_bicubic_224_FP32_minus1_plus1', image_cache_recipe=cache_recipe(cfg),
                    future_image_weight=1.0, source_fingerprint=contract_hash(sources))
    assets = {}
    if oat:
        contract.update(version=2, codec='oat', action_token_count=1920, action_token_length=16,
                        action_normalization='oat_min_max_gripper_1_minus_2g')
    codec_assets = ((cfg.framework.action_model.oat_tokenizer_path, ['config.json', 'model.safetensors', 'norm_stats.pt', 'policy_io.json'])
                    if oat else (cfg.framework.action_model.fast_tokenizer_name, ['processor_config.json', 'tokenizer.json']))
    for folder, names in [(cfg.framework.u0.vision_tokenizer, ['config.yaml', 'model.ckpt']), codec_assets]:
        for name in names:
            p = Path(folder) / name
            with p.open('rb') as f:
                assets[str(p.resolve())] = hashlib.file_digest(f, 'sha256').hexdigest()
    contract['encoder_sha256'] = assets
    cfg.framework.u0.sequence_contract = contract
    cfg.framework.u0.sequence_contract_sha256 = contract_hash(contract)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output)
    (output.parent / 'sequence_contract.json').write_text(json.dumps(contract, indent=2))
    (output.parent / 'sequence_windows.json').write_text(json.dumps(windows))
    print(f'States: {len(states)}, h={h}, valid windows: {sum(e[2] for rows in windows.values() for e in rows)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--h', type=int, default=2)
    args = parser.parse_args()
    if args.h < 1:
        parser.error('h must be positive')
    prepare(args.config, args.output, args.h)
