"""Prepare, train and evaluate GroundingPI on the shared LIBERO LeRobot v3 data."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
ENVS = Path('/share/project/robot-rfm/robot-exp/usr/cjy/envs')
RAW = Path('/share/project/robot-rfm/robot-raw-data/LIBERO_LeRobot_v3')
DATA = ROOT / 'playground/Datasets/LIBERO_LeRobot_v3_groundingpi'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90')
FRAMEWORKS = {'pi': 'groundingpipi', 'pi_v3': 'groundingpipi_v3', 'pi05': 'groundingpipi05', 'oft': 'groundingpioft', 'fast': 'groundingpifast'}


def environment():
    env = os.environ.copy()
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    for key, value in dict(WANDB_MODE='disabled', NO_ALBUMENTATIONS_UPDATE='1',
                           TOKENIZERS_PARALLELISM='false', TORCHDYNAMO_DISABLE='1',
                           OMP_NUM_THREADS='4', MAX_JOBS='4').items():
        env.setdefault(key, value)
    return env


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    source, dest = args.raw_data.resolve(), args.data_root.resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError('Raw data and writable adapter must be separate directories')
    suites = SUITES if args.mix in ('five', 'libero_90') else SUITES[:4]
    modality = json.loads((HERE / 'modality.json').read_text())
    # Explicit keys avoid relying on loader defaults for v3 columns.
    for fields, original in ((modality['state'], 'observation.state'), (modality['action'], 'action')):
        for field in fields.values():
            field['original_key'] = original
    manifest = {'source': str(source), 'suites': []}
    for suite in suites:
        src, dst = source / suite, dest / suite
        info = json.loads((src / 'meta/info.json').read_text())
        assert info['codebase_version'] == 'v3.0', info['codebase_version']
        assert info['features']['observation.state']['shape'] == [8]
        assert info['features']['action']['shape'] == [7]
        dst.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or (dst / 'meta').is_symlink():
            raise ValueError(f'Adapter metadata must be writable local files: {dst}')
        for name in ('data', 'videos'):
            link = dst / name
            if link.is_symlink():
                if link.resolve() != (src / name).resolve():
                    raise ValueError(f'Conflicting data link: {link}')
            elif link.exists():
                raise ValueError(f'Expected symlink, found existing directory: {link}')
            else:
                link.symlink_to(src / name, target_is_directory=True)
        meta = dst / 'meta'
        meta.mkdir(exist_ok=True)
        for original in (src / 'meta').rglob('*'):
            if original.is_file() and original.name not in ('modality.json', 'stats_gr00t.json'):
                copied = meta / original.relative_to(src / 'meta')
                copied.parent.mkdir(parents=True, exist_ok=True)
                if not copied.exists():
                    shutil.copy2(original, copied)
        write_json(meta / 'modality.json', modality)
        manifest['suites'].append({'name': suite, 'episodes': info['total_episodes'], 'frames': info['total_frames']})
    write_json(dest / 'groundingpi_manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def mix_name(mix):
    return 'libero_v3_' + mix if mix in ('four', 'five') else mix + '_v3'


def run_logged(command, log, env):
    print('Running:', ' '.join(map(str, command)), '\nLog:', log, flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as handle:
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def train(args):
    import torch
    from omegaconf import OmegaConf
    if importlib.util.find_spec('deepspeed') is None:
        raise RuntimeError('Install DeepSpeed in the starvla environment; see README_groundingpi.md')
    count = torch.cuda.device_count()
    gpus = args.gpus or count
    if gpus < 1 or gpus > count:
        raise ValueError(f'Requested {gpus} GPUs; visible: {count}')
    prepare(args)
    output = args.output.resolve()
    if (output / 'config.full.yaml').exists():
        raise FileExistsError(f'Refusing to overwrite an existing training run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(HERE / (FRAMEWORKS[args.framework] + '.yaml'))
    cfg.run_root_dir, cfg.run_id = str(output.parent), output.name
    cfg.framework.qwenvl.base_vlm = str(args.model.resolve())
    cfg.framework.action_model.fast_tokenizer = str(args.fast_tokenizer.resolve())
    if not (args.model / 'config.json').is_file():
        raise FileNotFoundError(args.model / 'config.json')
    if args.framework == 'fast' and not args.fast_tokenizer.is_dir():
        raise FileNotFoundError(args.fast_tokenizer)
    cfg.datasets.vla_data.data_root_dir = str(args.data_root.resolve())
    if args.framework == 'pi05':
        cfg.datasets.vla_data.data_mix = 'groundingpi_pi05_' + args.mix
    elif args.framework == 'pi_v3':
        if args.mix != 'libero_goal':
            raise ValueError('pi_v3 currently requires --mix libero_goal')
        cfg.datasets.vla_data.data_mix = 'groundingpi_v3_libero_goal'
    else:
        cfg.datasets.vla_data.data_mix = mix_name(args.mix)
    cfg.datasets.vla_data.per_device_batch_size = args.batch_size
    cfg.datasets.vla_data.num_workers = args.workers
    cfg.trainer.max_train_steps = args.steps
    if args.framework != 'pi_v3':
        cfg.trainer.num_warmup_steps = min(100, max(1, args.steps // 10))
    cfg.trainer.save_interval = args.save_interval
    cfg.trainer.eval_interval = args.steps + 1
    cfg.trainer.logging_frequency = min(10, args.steps)
    config_path = output / 'launch.yaml'
    OmegaConf.save(cfg, config_path)
    # Warm caches before distributed ranks start; also validates all selected suites.
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    dataset = get_vla_dataset(cfg.datasets.vla_data, balance_dataset_weights=True)
    for subset in dataset.datasets:
        for index in (0, len(subset) - 1):
            sample = subset[index]
            assert sample['action'].shape == (16, 7), sample['action'].shape
            assert len(sample['image']) == 2
            assert all(image.size == (224, 224) for image in sample['image'])
            if args.framework in ('pi', 'pi_v3', 'pi05'):
                assert sample['state'].shape == (1, 8)
    del dataset, subset, sample
    offload = args.cpu_offload or gpus == 1
    ds = json.loads((HERE / ('groundingpi_zero2_offload.json' if offload else 'groundingpi_zero2.json')).read_text())
    ds_path = output / 'deepspeed.json'
    write_json(ds_path, ds)
    accelerate = {'compute_environment': 'LOCAL_MACHINE', 'distributed_type': 'DEEPSPEED',
                  'num_machines': 1, 'num_processes': gpus,
                  'deepspeed_config': {'deepspeed_config_file': str(ds_path), 'zero3_init_flag': False}}
    launch_path = output / 'accelerate.yaml'
    OmegaConf.save(OmegaConf.create(accelerate), launch_path)
    env = environment()
    env.pop('STARVLA_DISABLE_DEEPSPEED', None)
    cuda_home = Path('/usr/local') / ('cuda-' + torch.version.cuda)
    if (cuda_home / 'bin/nvcc').is_file():
        env.setdefault('CUDA_HOME', str(cuda_home))
    command = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file', launch_path,
               '--num_processes', str(gpus), '--main_process_port', str(args.train_port),
               ROOT / 'starVLA/training/train_starvla.py', '--config_yaml', config_path]
    run_logged(command, output / 'train.log', env)
    checkpoint = output / 'final_model/pytorch_model.pt'
    if not checkpoint.is_file():
        raise RuntimeError(f'Training returned without final checkpoint: {checkpoint}')
    metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    write_json(output / 'train_result.json', {'framework': args.framework, 'steps': args.steps,
               'gpus': gpus, 'cpu_offload': offload, 'checkpoint': str(checkpoint),
               'first_metrics': metrics[0], 'last_metrics': metrics[-1]})
    return checkpoint


def evaluation_config(checkpoint, cfg):
    """Restore observation fields omitted by the accessed-only config snapshot."""
    from omegaconf import OmegaConf
    full = Path(checkpoint).parents[1] / 'config.full.yaml'
    if full.is_file():
        cfg = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(full), OmegaConf.create(cfg)), resolve=True)
    if cfg['framework']['name'] == 'GroundingPIPI' and 'include_state' not in cfg['datasets']['vla_data']:
        raise ValueError('PI evaluation requires an explicit training include_state setting')
    return cfg


def evaluate(args):
    import contextlib
    import torch
    from starVLA.model.framework.share_tools import read_mode_config
    from examples.simBenchmarks.LIBERO.eval_files.eval_sharding import summarize_records

    checkpoint = args.checkpoint.resolve()
    cfg, stats = read_mode_config(str(checkpoint))
    cfg = evaluation_config(checkpoint, cfg)
    if not args.libero_python.is_file():
        raise FileNotFoundError(args.libero_python)
    keys = list(stats)
    unnorm = args.unnorm_key or (keys[0] if len(keys) == 1 else None)
    if unnorm not in keys:
        raise ValueError(f'Specify --unnorm-key from {keys}')
    count = getattr(args, 'eval_gpus', 1)
    if count < 1 or count > torch.cuda.device_count():
        raise ValueError(f'Requested {count} eval GPUs; visible: {torch.cuda.device_count()}')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else list(map(str, range(torch.cuda.device_count())))
    output = args.eval_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'evaluation_manifest.json').exists():
        raise FileExistsError(f'Refusing to overwrite evaluation: {output}')
    for port in range(args.port, args.port + count):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    benchmark_root = args.libero_root.resolve() / 'libero/libero'
    config = {'benchmark_root': str(benchmark_root),
              'bddl_files': str(benchmark_root / 'bddl_files'),
              'init_states': str(benchmark_root / 'init_files'),
              'assets': str(benchmark_root / 'assets'), 'datasets': str(args.raw_data.resolve())}
    for key in ('bddl_files', 'init_states', 'assets'):
        if not Path(config[key]).is_dir():
            raise FileNotFoundError(config[key])
    write_json(output / 'evaluation_manifest.json',
               dict(checkpoint=str(checkpoint), suites=args.suites, trials_per_task=args.trials,
                    max_tasks=args.max_tasks, seed=args.seed, eval_gpus=count,
                    devices=devices[:count], ports=list(range(args.port, args.port + count)),
                    render_backend=os.environ.get('MUJOCO_GL', 'egl'),
                    unnorm_key=unnorm, include_state=cfg['datasets']['vla_data'].get('include_state', False),
                    image_resample='bicubic', sharding='(task_id * trials + episode_idx) % num_shards'))
    processes, workers, results = [], [], []
    with contextlib.ExitStack() as stack:
        def launch(command, log_path, env):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(log_path.open('w'))
            proc = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                    stdout=handle, stderr=subprocess.STDOUT)
            processes.append(proc)
            return proc

        try:
            for shard in range(count):
                worker_out = output if count == 1 else output / f'shard_{shard}'
                env = environment()
                env.update(CUDA_VISIBLE_DEVICES=devices[shard], TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
                env.setdefault('MUJOCO_GL', 'egl')
                env.setdefault('PYOPENGL_PLATFORM', env['MUJOCO_GL'])
                # robosuite maps the physical CUDA_VISIBLE_DEVICES entry to EGL.
                env.pop('MUJOCO_EGL_DEVICE_ID', None)
                env['PYTHONPATH'] = str(args.libero_root.resolve()) + os.pathsep + env['PYTHONPATH']
                config_dir = worker_out / 'libero_config'
                write_json(config_dir / 'config.yaml', config)
                env['LIBERO_CONFIG_PATH'] = str(config_dir)
                port = args.port + shard
                command = [sys.executable, ROOT / 'deployment/model_server/server_policy.py',
                           '--ckpt_path', checkpoint, '--port', str(port), '--seed', str(args.seed)]
                for key in ('include_state', 'obs_image_size', 'normalize_state_on_server'):
                    if key in cfg['datasets']['vla_data']:
                        command += ['--config_override', f'datasets.vla_data.{key}=' + json.dumps(cfg['datasets']['vla_data'][key])]
                server = launch(command, worker_out / 'server.log', env)
                workers.append((shard, worker_out, env, port, server))
            deadline = time.monotonic() + 600
            pending = set(range(count))
            while pending:
                for shard, worker_out, env, port, server in workers:
                    if server.poll() is not None:
                        raise RuntimeError(f'Policy server exited: {worker_out / "server.log"}')
                    if shard in pending:
                        try:
                            with socket.create_connection(('127.0.0.1', port), timeout=1):
                                pending.remove(shard)
                        except OSError:
                            pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Policy server startup exceeded 600 seconds: {pending}')
                if pending:
                    time.sleep(1)
            for suite in args.suites:
                clients = []
                for shard, worker_out, env, port, server in workers:
                    suite_out = worker_out / suite
                    command = [args.libero_python, ROOT / 'examples/simBenchmarks/LIBERO/eval_files/eval_libero.py',
                               '--args.port', str(port), '--args.task-suite-name', suite,
                               '--args.image-resample', 'bicubic', '--args.num-trials-per-task', str(args.trials),
                               '--args.max-tasks', str(args.max_tasks), '--args.video-out-path', suite_out,
                               '--args.unnorm-key', unnorm, '--args.seed', str(args.seed),
                               '--args.num-shards', str(count), '--args.shard-id', str(shard)]
                    if cfg['datasets']['vla_data'].get('include_state', False):
                        command.append('--args.include-state')
                    clients.append((launch(command, suite_out / 'eval.log', env), suite_out))
                while True:
                    for _, worker_out, _, _, server in workers:
                        if server.poll() is not None:
                            raise RuntimeError(f'Policy server exited: {worker_out / "server.log"}')
                    for client, suite_out in clients:
                        if client.poll() not in (None, 0):
                            raise RuntimeError(f'Evaluation failed: {suite_out / "eval.log"}')
                    if all(client.poll() == 0 for client, _ in clients):
                        break
                    time.sleep(2)
                shards = [json.loads((path / 'result.json').read_text()) for _, path in clients]
                num_tasks = shards[0]['num_tasks']
                if any(s['num_tasks'] != num_tasks or s['trials_per_task'] != args.trials for s in shards):
                    raise ValueError('Shard evaluation configurations disagree')
                records = [r for s in shards for r in s['episode_records']]
                result = summarize_records(records, num_tasks, args.trials)
                result.update(suite=suite, seed=args.seed, num_tasks=num_tasks,
                              trials_per_task=args.trials, eval_gpus=count,
                              include_state=cfg['datasets']['vla_data'].get('include_state', False))
                write_json(output / suite / 'result.json', result)
                results.append(result)
            write_json(output / 'result.json', {'checkpoint': str(checkpoint), 'status': 'complete', 'suites': results})
        except BaseException as exc:
            write_json(output / 'result.json', {'checkpoint': str(checkpoint), 'status': 'incomplete',
                                               'suites': results, 'error': str(exc)})
            raise
        finally:
            for proc in reversed(processes):
                if proc.poll() is None:
                    proc.terminate()
            deadline = time.monotonic() + 15
            for proc in reversed(processes):
                try:
                    proc.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def smoke(args):
    output = args.output.resolve()
    results = []
    for framework in FRAMEWORKS:
        local = argparse.Namespace(**vars(args))
        local.framework = framework
        local.output = output / framework
        local.gpus, local.cpu_offload = 1, True
        local.batch_size, local.workers = 1, 0
        local.save_interval = args.steps + 1
        local.suites, local.max_tasks, local.trials = ['libero_spatial'], 1, 1
        local.eval_output = local.output / 'evaluation'
        record = {'framework': framework, 'training': 'pending', 'evaluation': 'pending'}
        try:
            local.checkpoint = train(local)
            record['training'] = 'passed'
            evaluate(local)
            record['evaluation'] = 'passed'
        except Exception as exc:
            record['error'] = str(exc)
            record['evaluation' if record['training'] == 'passed' else 'training'] = 'failed'
        results.append(record)
        write_json(output / 'smoke_result.json', results)
    if any(row.get('error') for row in results):
        raise RuntimeError(f'Smoke failures recorded in {output / "smoke_result.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'eval', 'smoke'))
    parser.add_argument('--framework', choices=FRAMEWORKS, default='pi')
    parser.add_argument('--raw-data', type=Path, default=RAW)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--mix', choices=('four', 'five') + SUITES, default='four')
    parser.add_argument('--model', type=Path, default=ROOT / 'playground/Pretrained_models/GroundingPI')
    parser.add_argument('--fast-tokenizer', type=Path, default=ROOT / 'playground/Pretrained_models/fast')
    parser.add_argument('--gpus', type=int, default=0, help='0: all visible GPUs')
    parser.add_argument('--cpu-offload', action='store_true', help='Always enabled on one GPU')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--train-port', type=int, default=29507)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--libero-python', type=Path, default=ENVS / 'libero/bin/python')
    parser.add_argument('--libero-root', type=Path,
                        default=ENVS.parent / 'simulation/LIBERO')
    parser.add_argument('--port', type=int, default=10093)
    parser.add_argument('--eval-gpus', type=int, default=1, help='Number of independent evaluation shards/GPUs')
    parser.add_argument('--unnorm-key')
    parser.add_argument('--suites', nargs='+', choices=SUITES, default=list(SUITES[:4]))
    parser.add_argument('--max-tasks', type=int, default=-1)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--eval-output', type=Path, default=None)
    args = parser.parse_args()
    args.steps = args.steps if args.steps is not None else (100 if args.command == 'smoke' else 10000)
    if args.steps < 1 or args.batch_size < 1 or args.save_interval < 1 or args.trials < 1:
        parser.error('steps, batch-size, save-interval and trials must be positive')
    if args.command == 'eval' and args.checkpoint is None:
        parser.error('eval requires --checkpoint')
    stamp = time.strftime('%Y%m%d_%H%M%S')
    args.output = args.output or ROOT / 'playground/Checkpoints' / f'groundingpi_{args.command}_{args.framework}_{stamp}'
    args.eval_output = args.eval_output or args.output / 'evaluation'
    os.environ.update(environment())
    {'prepare': prepare, 'train': train, 'eval': evaluate, 'smoke': smoke}[args.command](args)


if __name__ == '__main__':
    main()
