"""Prepare/validate RoboTwin Clean v3; training requires an explicit train command."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
RAW = Path('/share/project/robot-rfm/robot-raw-data/robotwin2.0-lerobot-v3.0')
DATA = ROOT / 'playground/Datasets/RoboTwin_clean_v3_groundingpi'
OUTPUT = ROOT / 'playground/Checkpoints/groundingpipi_v3_robotwin_clean_8gpu_b16_10k'
CAMERAS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
FIELDS = [('left_joints', 0, 6), ('right_joints', 7, 13),
          ('left_gripper', 6, 7), ('right_gripper', 13, 14)]
ORDER = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def modality():
    return {
        **{kind: {name: {'start': start, 'end': end,
                         'original_key': 'action' if kind == 'action' else 'observation.state',
                         'dtype': 'float32', 'absolute': True}
                  for name, start, end in FIELDS} for kind in ('state', 'action')},
        'video': {cam: {'original_key': 'observation.images.' + cam} for cam in CAMERAS},
        'annotation': {'human.action.task_description': {'original_key': 'task_index'}},
    }


def metadata_digest(source):
    digest = hashlib.sha256()
    for path in sorted((source / 'meta').rglob('*')):
        if path.is_file():
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(args):
    if getattr(args, 'framework', None) == 'pi05_eef':
        from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_dataset import prepare as prepare_eef
        if getattr(args, 'dataset_root', None) is not None:
            from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_lerobot_dataset import prepare as prepare_lerobot
            return prepare_lerobot(args.dataset_root, args.data_root,
                                   args.cot_root if args.with_cot else None,
                                   image_input_mode=getattr(args, 'image_input_mode', 'composite'),
                                   instruction_source=getattr(args, 'instruction_source', 'default'))
        return prepare_eef(args.raw_data, args.data_root)
    source, dest = (args.raw_data / 'clean').resolve(), args.data_root.resolve()
    raw = args.raw_data.resolve()
    if dest == raw or dest in raw.parents or raw in dest.parents:
        raise ValueError('Raw data and writable adapter must be separate directories')
    info = json.loads((source / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0' or info['fps'] != 50:
        raise ValueError('Expected RoboTwin Clean v3.0 at 50 FPS')
    for key in ('action', 'observation.state'):
        if info['features'][key]['shape'] != [14]:
            raise ValueError(f'Expected 14-D {key}')
    manifest = {'source': str(source), 'source_metadata_sha256': metadata_digest(source),
                'lerobot_version': 'v3.0', 'fps': 50, 'episodes': info['total_episodes'],
                'frames': info['total_frames'], 'action_horizon': 16,
                'camera_order': CAMERAS, 'packed_to_raw_indices': ORDER,
                'fields': [name for name, _, _ in FIELDS],
                'modality': modality(), 'action_mode': 'abs', 'include_state': True}
    manifest_path = dest / 'adapter_manifest.json'
    if dest.is_symlink() or (dest / 'clean').is_symlink() or (dest / 'clean/meta').is_symlink():
        raise ValueError('Adapter and metadata must be local directories')
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Adapter source/contract changed; use a new data-root')
    for name in ('data', 'videos'):
        link, target = dest / 'clean' / name, source / name
        if not target.is_dir():
            raise FileNotFoundError(target)
        if link.is_symlink():
            if link.resolve() != target:
                raise ValueError(f'Conflicting link: {link}')
        elif link.exists():
            raise ValueError(f'Expected a data symlink: {link}')
    meta = dest / 'clean/meta'
    meta.mkdir(parents=True, exist_ok=True)
    for path in (source / 'meta').rglob('*'):
        if path.is_file() and path.name not in ('modality.json', 'stats_gr00t.json'):
            copied = meta / path.relative_to(source / 'meta')
            copied.parent.mkdir(parents=True, exist_ok=True)
            if copied.is_symlink():
                raise ValueError(f'Metadata cannot be a symlink: {copied}')
            if copied.exists() and copied.read_bytes() != path.read_bytes():
                raise ValueError(f'Conflicting metadata: {copied}')
            if not copied.exists():
                shutil.copy2(path, copied)
    for name in ('data', 'videos'):
        link = dest / 'clean' / name
        if not link.is_symlink():
            link.symlink_to(source / name, target_is_directory=True)
    write_json(meta / 'modality.json', modality())
    write_json(manifest_path, manifest)
    print(f'Prepared {info["total_episodes"]} clean episodes at {dest}', flush=True)
    return manifest


def config(args):
    from omegaconf import OmegaConf
    framework = getattr(args, 'framework', 'pi_v3')
    cfg = OmegaConf.load(HERE / ({'pi05': 'groundingpipi05_clean.yaml', 'pi05_eef': 'groundingpipi05_eef_clean.yaml'}.get(framework, 'groundingpipi_v3_clean.yaml')))
    if getattr(args, "knowledge_insulation", False):
        if framework != "pi05":
            raise ValueError("Knowledge insulation requires pi05")
        cfg.framework.knowledge_insulation = True
        cfg.framework.action_model.fast_tokenizer = str(ROOT / "playground/Pretrained_models/fast")
    cfg.framework.qwenvl.base_vlm = str(args.model.resolve())
    cfg.datasets.vla_data.data_root_dir = str(args.data_root.resolve())
    cfg.datasets.vla_data.per_device_batch_size = args.batch_size
    cfg.datasets.vla_data.num_workers = args.workers
    size = getattr(args, 'image_size', 224)
    cfg.datasets.vla_data.image_size = [size, size]
    cfg.datasets.vla_data.obs_image_size = [size, size]
    if framework == 'pi05_eef':
        cfg.datasets.vla_data.image_size = [384, 320]
        cfg.datasets.vla_data.obs_image_size = [384, 320]
        cfg.datasets.vla_data.raw_data_dir = str(args.raw_data.resolve())
        cfg.datasets.vla_data.color_jitter = getattr(args, 'color_jitter', 0.0)
        cfg.datasets.vla_data.filter_static_segments = getattr(args, 'filter_static_segments', False)
    if getattr(args, 'dataset_root', None) is not None:
        del cfg.datasets.vla_data.raw_data_dir
        cfg.datasets.vla_data.dataset_py = 'robotwin_eef_lerobot'
        cfg.datasets.vla_data.dataset_root = str(args.dataset_root.resolve())
        cfg.datasets.vla_data.quaternion_order = 'wxyz'
        cfg.datasets.vla_data.with_cot = args.with_cot
        cfg.datasets.vla_data.instruction_source = getattr(args, 'instruction_source', 'default')
        cfg.datasets.vla_data.cot_root = str(args.cot_root.resolve()) if args.with_cot else None
        cfg.datasets.vla_data.image_input_mode = getattr(args, 'image_input_mode', 'composite')
        if cfg.datasets.vla_data.image_input_mode == 'separate':
            cfg.datasets.vla_data.image_size = [240, 320]
            cfg.datasets.vla_data.obs_image_size = [240, 320]
            cfg.datasets.vla_data.image_layout = 'head_left_wrist_right_separate'
        cfg.framework.cot_loss_weight = args.cot_loss_weight if args.with_cot else 0.0
        if args.with_cot:
            from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_lerobot_dataset import (
                COMPOSITE_COT_PROMPT, SEPARATE_COT_PROMPT,
            )
            separate = cfg.datasets.vla_data.image_input_mode == 'separate'
            cfg.framework.cot_training_mode = 'separate_head_v1' if separate else 'shared_canvas_v1'
            cfg.framework.cot_prompt = SEPARATE_COT_PROMPT if separate else COMPOSITE_COT_PROMPT
    cfg.trainer.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    cfg.trainer.max_train_steps = args.steps
    for name in ('num_warmup_steps', 'health_check_interval', 'save_interval'):
        value = getattr(args, name, None)
        if value is not None:
            cfg.trainer[name] = value
    cfg.trainer.logging_frequency = min(10, args.steps)
    cfg.run_root_dir, cfg.run_id = str(args.output.resolve().parent), args.output.name
    return cfg


def validation_path(args):
    size = getattr(args, 'image_size', 224)
    framework = getattr(args, 'framework', 'pi_v3')
    if framework == 'pi05_eef':
        return args.data_root / ('validation_report_pi05_eef_lerobot.json' if getattr(args, 'dataset_root', None) is not None else 'validation_report_pi05_eef.json')
    if framework == 'pi05':
        return args.data_root / f'validation_report_pi05_{size}.json'
    return args.data_root / ('validation_report.json' if size == 224 else f'validation_report_{size}.json')


def check_tables(source):
    """Check every frame, episode, language reference and physical shard path."""
    import numpy as np
    import pandas as pd
    info = json.loads((source / 'meta/info.json').read_text())
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((source / 'meta/episodes').glob('*/*.parquet'))])
    episodes = episodes.sort_values('episode_index').set_index('episode_index')
    if episodes.index.has_duplicates or list(episodes.index) != list(range(info['total_episodes'])):
        raise ValueError('Episode IDs must be unique and contiguous')
    if int(episodes['length'].sum()) != info['total_frames']:
        raise ValueError('Episode lengths disagree with total_frames')
    tasks = pd.read_parquet(source / 'meta/tasks.parquet')
    if 'task' not in tasks.columns:
        tasks = tasks.rename_axis('task').reset_index()
    if not {'task_index', 'task'}.issubset(tasks.columns):
        raise ValueError('Expected task_index and language text')
    if tasks['task_index'].duplicated().any() or not tasks['task'].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError('Invalid task IDs or language text')
    expected_files = {source / info['data_path'].format(chunk_index=int(row['data/chunk_index']),
                     file_index=int(row['data/file_index'])) for _, row in episodes.iterrows()}
    if expected_files != set(source.glob('data/*/*.parquet')):
        raise ValueError('Physical data shards differ from episode metadata')
    seen, frame_count = set(), 0
    boundaries, videos = set(), set()
    for path in sorted(expected_files):
        print(f'Checking {path.name}', flush=True)
        frame = pd.read_parquet(path)
        frame_count += len(frame)
        if not frame['task_index'].isin(tasks['task_index']).all():
            raise ValueError(f'Unknown language indices: {path}')
        for key in ('observation.state', 'action'):
            values = np.stack(frame[key])
            if values.shape != (len(frame), 14) or not np.isfinite(values).all():
                raise ValueError(f'Invalid numeric data: {path} {key}')
        for episode_id, rows in frame.groupby('episode_index', sort=False):
            if episode_id in seen or episode_id not in episodes.index:
                raise ValueError(f'Duplicate/unexpected episode {episode_id}')
            seen.add(episode_id)
            ep = episodes.loc[episode_id]
            if len(rows) != int(ep['length']):
                raise ValueError(f'Episode {episode_id} length mismatch')
            expected_path = source / info['data_path'].format(chunk_index=int(ep['data/chunk_index']), file_index=int(ep['data/file_index']))
            if expected_path != path:
                raise ValueError(f'Episode {episode_id} points to wrong data shard')
            np.testing.assert_array_equal(rows['frame_index'], np.arange(len(rows)))
            np.testing.assert_array_equal(rows['index'], np.arange(int(ep['dataset_from_index']), int(ep['dataset_to_index'])))
            np.testing.assert_allclose(rows['timestamp'], np.arange(len(rows))/50, atol=1e-5, rtol=1e-6)
            for cam in CAMERAS:
                prefix = 'videos/observation.images.' + cam
                video = source / info['video_path'].format(video_key='observation.images.'+cam,
                    chunk_index=int(ep[prefix+'/chunk_index']), file_index=int(ep[prefix+'/file_index']))
                if not video.is_file():
                    raise FileNotFoundError(video)
                start, end = float(ep[prefix+'/from_timestamp']), float(ep[prefix+'/to_timestamp'])
                if start < 0 or end <= start or abs((end-start)-len(rows)/50) > .04:
                    raise ValueError(f'Bad video timestamps: episode {episode_id} {cam}')
                videos.add(video)
        ids = frame['episode_index'].unique()
        boundaries.update((int(ids[0]), int(ids[-1])))
    if seen != set(episodes.index) or frame_count != info['total_frames']:
        raise ValueError('Missing episodes or frames')
    for cam in CAMERAS:
        prefix = 'videos/observation.images.' + cam
        for _, group in episodes.groupby([prefix+'/chunk_index', prefix+'/file_index']):
            boundaries.update((int(group.index.min()), int(group.index.max())))
    return episodes, boundaries, {'episodes': len(episodes), 'frames': frame_count,
        'language_variants': len(tasks), 'data_files': len(expected_files), 'video_files': len(videos)}


def validate(args):
    if getattr(args, 'framework', None) == 'pi05_eef':
        from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_validation import validate as validate_eef
        return validate_eef(args)
    import av
    import numpy as np
    from PIL import Image
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps
    manifest = prepare(args)
    report_path = validation_path(args)
    size = getattr(args, 'image_size', 224)
    # A failed rerun must not leave an older successful report usable for launch.
    write_json(report_path, {'passed': False, 'manifest': manifest})
    episodes, boundaries, counts = check_tables(Path(manifest['source']))
    # Check stream timing independently of the frame-table timestamps.
    info = json.loads((Path(manifest['source']) / 'meta/info.json').read_text())
    checked_videos = set()
    for _, ep in episodes.iterrows():
        for cam in CAMERAS:
            original = 'observation.images.' + cam
            prefix = 'videos/' + original
            path = Path(manifest['source']) / info['video_path'].format(video_key=original,
                chunk_index=int(ep[prefix+'/chunk_index']), file_index=int(ep[prefix+'/file_index']))
            if path not in checked_videos:
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    if stream.average_rate != 50 or (stream.height, stream.width) != (480, 640):
                        raise ValueError(f'Unexpected video FPS/resolution: {path}')
                checked_videos.add(path)
    cfg = config(args)
    dataset = get_vla_dataset(cfg.datasets.vla_data)
    if len(dataset.datasets) != 1 or dataset.datasets[0].dataset_name != 'clean':
        raise ValueError('Expected exactly one Clean dataset')
    subset = dataset.datasets[0]
    if len(subset) != counts['frames'] or len(subset.trajectory_ids) != counts['episodes']:
        raise ValueError('Loader skipped data')
    chosen = boundaries | set(map(int, np.linspace(0, len(episodes)-1, args.sample_episodes)))
    # Original conversion concatenates the 50 demonstrations of each task.
    chosen.update(range(0, len(episodes), 50))
    checks = []
    for position, ep_id in enumerate(sorted(chosen)):
        ep = episodes.loc[ep_id]
        for step in sorted({0, int(ep['length'])-1}):
            raw = subset.get_step_data(ep_id, step)
            # Compare slices directly against unmodified 14-D source arrays.
            frame = subset.curr_traj_data
            for kind, original in [('state', 'observation.state'), ('action', 'action')]:
                indices = np.minimum(np.arange(step, step+(16 if kind == 'action' else 1)), len(frame)-1)
                expected = np.stack(frame[original].iloc[indices])[:, ORDER]
                actual = np.concatenate([raw[key] for key in subset.modality_keys[kind]], axis=-1)
                np.testing.assert_array_equal(actual, expected)
            expected_language = subset.tasks.loc[int(frame['task_index'].iloc[step]), 'task']
            if raw[subset.modality_keys['language'][0]][0] != expected_language:
                raise ValueError('Language lookup disagrees with source')
            data = subset.transforms(raw)
            sample = subset._pack_sample(data)
            if sample['state'].shape != (1,14) or sample['action'].shape != (16,14):
                raise ValueError('Invalid sample shape')
            for key in ('state', 'action'):
                if not np.isfinite(sample[key]).all() or np.max(np.abs(sample[key])) > 1.0001:
                    raise ValueError(f'Invalid normalized {key}')
            if len(sample['image']) != 3 or not all(im.size == (size,size) for im in sample['image']):
                raise ValueError('Invalid camera count/resolution')
            for cam, packed in zip(CAMERAS, sample['image']):
                prefix = 'videos/observation.images.'+cam
                path = Path(manifest['source']) / subset.video_path_pattern.format(video_key='observation.images.'+cam,
                    chunk_index=int(ep[prefix+'/chunk_index']), file_index=int(ep[prefix+'/file_index']))
                time = float(frame['timestamp'].iloc[step]) + float(ep[prefix+'/from_timestamp'])
                decoded = get_frames_by_timestamps(str(path), np.array([time]), video_backend='pyav')[0]
                np.testing.assert_array_equal(np.asarray(packed), np.asarray(Image.fromarray(decoded).resize((size,size))))
                if decoded.std() == 0:
                    raise ValueError(f'Uniform camera frame: episode {ep_id} {cam}')
            checks.append({'episode': ep_id, 'step': step, 'language': sample['lang']})
        if position % 10 == 0:
            print(f'Decoded {position+1}/{len(chosen)} selected episodes', flush=True)
    dataset.save_dataset_statistics(args.data_root / 'dataset_statistics.json')
    if metadata_digest(Path(manifest['source'])) != manifest['source_metadata_sha256']:
        raise ValueError('Raw metadata changed during validation')
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    image_inputs = processor.image_processor(images=[Image.new('RGB', (size,size)) for _ in CAMERAS], return_tensors='pt')
    token_counts = [int(grid.prod()) // processor.image_processor.merge_size**2 for grid in image_inputs['image_grid_thw']]
    expected_tokens = (size // 28)**2
    if token_counts != [expected_tokens]*3:
        raise ValueError(f'Unexpected image token counts: {token_counts}; check IMAGE_MAX_TOKEN_NUM')
    report = {'passed': True, 'manifest': manifest, **counts, 'decoded_episodes': len(chosen),
              'samples': checks, 'model_contract': {'framework': cfg.framework.name,
              'state_shape': [1,14], 'action_shape': [16,14], 'image_count': 3, 'image_size': [size,size],
              'image_tokens_per_view': token_counts},
              'training_started': False}
    write_json(report_path, report)
    print(json.dumps({key: value for key,value in report.items() if key not in ('manifest','samples')}, indent=2), flush=True)
    return report


def train(args):
    from omegaconf import OmegaConf
    manifest = prepare(args)
    report_path = validation_path(args)
    report = json.loads(report_path.read_text())
    if not report.get('passed') or report.get('manifest') != manifest:
        raise ValueError('Run validate successfully for this adapter before preparing a launch')
    size = getattr(args, 'image_size', 224)
    separate = getattr(args, 'image_input_mode', 'composite') == 'separate'
    expected_size = ([240, 320] if separate else
                     [384, 320] if getattr(args, 'framework', None) == 'pi05_eef' else [size, size])
    if report.get('model_contract', {}).get('image_size') != expected_size:
        raise ValueError('Run validate at the requested image size before preparing a launch')
    expected_count = 3 if separate else 1
    if report.get('model_contract', {}).get('image_count') != expected_count:
        raise ValueError('Run validate with the requested image input mode before preparing a launch')
    if not (args.model / 'config.json').is_file():
        raise FileNotFoundError(args.model / 'config.json')
    output = args.output.resolve()
    # Dry-run artifacts can be reused, but an actual run can never be overwritten.
    if output.exists() and any(p.name not in {'launch.yaml','deepspeed.json','accelerate.yaml','launch_command.sh','dataset_statistics.json'} for p in output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    cfg = config(args)
    OmegaConf.save(cfg, output / 'launch.yaml')
    ds = {'fp16': {'enabled': False}, 'bf16': {'enabled': False},
          'train_micro_batch_size_per_gpu': 'auto', 'train_batch_size': 'auto',
          'gradient_accumulation_steps': cfg.trainer.gradient_accumulation_steps,
          'gradient_clipping': 1.0, 'steps_per_print': 10,
          'zero_optimization': {'stage': 2, 'allgather_partitions': True,
              'allgather_bucket_size': 50000000, 'reduce_scatter': True,
              'reduce_bucket_size': 50000000, 'overlap_comm': True, 'contiguous_gradients': True}}
    write_json(output / 'deepspeed.json', ds)
    accelerate = {'compute_environment': 'LOCAL_MACHINE', 'distributed_type': 'DEEPSPEED',
        'num_machines': 1, 'num_processes': args.gpus,
        'deepspeed_config': {'deepspeed_config_file': str(output / 'deepspeed.json'), 'zero3_init_flag': False}}
    OmegaConf.save(OmegaConf.create(accelerate), output / 'accelerate.yaml')
    shutil.copy2(args.data_root / 'dataset_statistics.json', output / 'dataset_statistics.json')
    command = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file', str(output / 'accelerate.yaml'),
               '--num_processes', str(args.gpus), '--main_process_port', str(args.train_port),
               str(ROOT / 'starVLA/training/train_starvla.py'), '--config_yaml', str(output / 'launch.yaml')]
    env = os.environ.copy()
    env.update(WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false', NO_ALBUMENTATIONS_UPDATE='1',
               OMP_NUM_THREADS='4', MAX_JOBS='4', TORCHDYNAMO_DISABLE='1')
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    env.pop('STARVLA_DISABLE_DEEPSPEED', None)
    import torch
    cuda_home = Path('/usr/local') / ('cuda-' + str(torch.version.cuda))
    if (cuda_home / 'bin/nvcc').is_file():
        env.setdefault('CUDA_HOME', str(cuda_home))
    env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(args.gpus))))
    script = '#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(ROOT)) + '\n'
    for key in ('WANDB_MODE','TOKENIZERS_PARALLELISM','NO_ALBUMENTATIONS_UPDATE','OMP_NUM_THREADS','MAX_JOBS',
                'TORCHDYNAMO_DISABLE','PYTHONPATH','CUDA_VISIBLE_DEVICES','CUDA_HOME'):
        if key in env:
            script += 'export ' + key + '=' + shlex.quote(env[key]) + '\n'
    # Re-enter the guarded runner so launching this file also checks provenance
    # and refuses to overwrite a previously started run.
    guarded = [sys.executable, str(Path(__file__).resolve()), 'train',
               '--raw-data', str(args.raw_data.resolve()), '--data-root', str(args.data_root.resolve()),
               '--output', str(output), '--model', str(args.model.resolve()),
               '--gpus', str(args.gpus), '--steps', str(args.steps), '--batch-size', str(args.batch_size),
               '--workers', str(args.workers), '--train-port', str(args.train_port),
               '--image-size', str(size), '--gradient-accumulation-steps', str(cfg.trainer.gradient_accumulation_steps),
               '--image-input-mode', getattr(args, 'image_input_mode', 'composite'),
               '--framework', getattr(args, 'framework', 'pi_v3')]
    if getattr(args, 'dataset_root', None) is not None:
        guarded += ['--dataset-root', str(args.dataset_root.resolve()), '--cot-root', str(args.cot_root.resolve()),
                    '--cot-loss-weight', str(args.cot_loss_weight),
                    '--instruction-source', getattr(args, 'instruction_source', 'default')]
        if args.with_cot:
            guarded.append('--with-cot')
    if getattr(args, 'knowledge_insulation', False):
        guarded.append('--knowledge-insulation')
    guarded += ['--color-jitter', str(getattr(args, 'color_jitter', 0.0))]
    if getattr(args, 'filter_static_segments', False):
        guarded.append('--filter-static-segments')
    for name in ('num_warmup_steps', 'health_check_interval', 'save_interval'):
        value = getattr(args, name, None)
        if value is not None:
            guarded += ['--' + name.replace('_', '-'), str(value)]
    script += 'unset STARVLA_DISABLE_DEEPSPEED\n' + shlex.join(guarded) + '\n'
    (output / 'launch_command.sh').write_text(script)
    print('Launch prepared:', output / 'launch_command.sh', flush=True)
    if args.dry_run:
        print('Dry run: no model loaded and no training process started.', flush=True)
        return
    if torch.cuda.device_count() < args.gpus:
        raise ValueError(f'Requested {args.gpus} GPUs; visible: {torch.cuda.device_count()}')
    with (output / 'train.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','validate','train'])
    parser.add_argument('--framework', choices=('pi_v3', 'pi05', 'pi05_eef'), default='pi_v3')
    parser.add_argument('--knowledge-insulation', action='store_true')
    parser.add_argument('--color-jitter', type=float, default=0.0)
    parser.add_argument('--filter-static-segments', action='store_true')
    parser.add_argument('--num-warmup-steps', type=int)
    parser.add_argument('--health-check-interval', type=int)
    parser.add_argument('--save-interval', type=int)
    parser.add_argument('--dataset-root', type=Path, help='Read EEF20 LeRobot v3 instead of HDF5')
    parser.add_argument('--with-cot', action='store_true')
    parser.add_argument('--image-input-mode', choices=('composite', 'separate'), default='composite')
    parser.add_argument('--cot-root', type=Path, default=Path(
        '/share/project/robot-rfm/robot-processed-data/lerobot_v3_data/robotwin2.0_Lerobotv3_cot/clean/20261029_v1'))
    parser.add_argument('--cot-loss-weight', type=float, default=1.0)
    parser.add_argument('--instruction-source', choices=('default', 'seen'), default='default')
    parser.add_argument('--raw-data', type=Path, default=RAW)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', type=Path, default=ROOT / 'playground/Pretrained_models/GroundingPI')
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, choices=(224,448), default=224)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--train-port', type=int, default=29527)
    parser.add_argument('--sample-episodes', type=int, default=12)
    parser.add_argument('--dry-run', action='store_true', help='Generate launch artifacts without starting training')
    args = parser.parse_args()
    import math
    if args.dataset_root is not None and args.framework != 'pi05_eef':
        parser.error('--dataset-root requires pi05_eef')
    if args.with_cot and args.dataset_root is None:
        parser.error('--with-cot requires --dataset-root')
    if args.instruction_source == 'seen' and args.dataset_root is None:
        parser.error('--instruction-source seen requires --dataset-root')
    if args.image_input_mode == 'separate' and (args.framework != 'pi05_eef' or args.dataset_root is None):
        parser.error('separate images require the pi05_eef LeRobot dataset')
    if not math.isfinite(args.cot_loss_weight) or args.cot_loss_weight <= 0:
        parser.error('--cot-loss-weight must be finite and positive')
    if not 0 <= args.color_jitter <= 1:
        parser.error('color-jitter must be in [0, 1]')
    if (args.color_jitter or args.filter_static_segments) and args.framework != 'pi05_eef':
        parser.error('Color jitter and static resampling require pi05_eef')
    if args.framework == 'pi05_eef':
        if args.raw_data == RAW:
            args.raw_data = Path('/share/project/robot-rfm/robot-raw-data/RoboTwin2.0/dataset')
        if args.data_root == DATA:
            suffix = ('lerobot_cot_separate_v1' if args.image_input_mode == 'separate' and args.with_cot else
                      'lerobot_separate_v1' if args.image_input_mode == 'separate' else
                      'lerobot_cot_canvas_v1' if args.with_cot else 'lerobot_v1')
            if args.instruction_source == 'seen':
                suffix = suffix.removesuffix('_v1') + '_seen100_v1'
            args.data_root = (ROOT / 'playground/Datasets' / ('robotwin_eef20_' + suffix) if args.dataset_root is not None
                              else Path('/share/project/robot-rfm/robot-processed-data/robotwin_eef20_h32_v1'))
    if args.output is None:
        if args.framework in ('pi05', 'pi05_eef'):
            label = ('groundingpipi05_eef20_h32_separate_cot_240x320' if args.image_input_mode == 'separate' and args.with_cot else
                     'groundingpipi05_eef20_h32_384x320' if args.framework == 'pi05_eef' else
                     'groundingpipi05_robotwin_clean')
            args.output = ROOT / 'playground/Checkpoints' / f'{label}_{time.strftime("%Y%m%d_%H%M%S")}'
        elif args.image_size == 224:
            args.output = OUTPUT
        else:
            args.output = ROOT / 'playground/Checkpoints' / (
                f'groundingpipi_v3_robotwin_clean_448_8gpu_{time.strftime("%Y%m%d_%H%M%S")}')
    if min(args.gpus,args.steps,args.batch_size,args.sample_episodes,args.gradient_accumulation_steps) < 1 or args.workers < 0:
        parser.error('Counts must be positive; workers must be nonnegative')
    if any(getattr(args, key) is not None and getattr(args, key) < 1 for key in ('num_warmup_steps', 'health_check_interval', 'save_interval')):
        parser.error('Warmup, health and save intervals must be positive')
    {'prepare': prepare, 'validate': validate, 'train': train}[args.command](args)


if __name__ == '__main__':
    main()
