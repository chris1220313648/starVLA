"""Sweep eight-GPU DataLoader concurrency, confirm the winners, then start a fresh run."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def progress_times(text):
    result = {}
    for match in re.finditer(r'(\d+)/200\s*\[((?:\d+:)?\d+:\d+)<', text):
        seconds = 0
        for part in match[2].split(':'):
            seconds = 60*seconds + int(part)
        result[int(match[1])] = seconds
    return result


def measured_seconds(text):
    progress = progress_times(text)
    if 50 not in progress or 190 not in progress or progress[190] <= progress[50]:
        raise ValueError('Missing valid steady-state timing boundaries at steps 50 and 190')
    return (progress[190]-progress[50])/140


def choose_workers(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result['workers'], []).append(result['seconds_per_step'])
    confirmed = {workers: sum(values)/len(values) for workers, values in grouped.items() if len(values) == 2}
    if len(confirmed) != 2:
        raise ValueError('Both finalists need two successful measurements')
    fastest = min(confirmed.values())
    return min(workers for workers, seconds in confirmed.items() if seconds <= fastest*1.05), confirmed


def gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    return [int(line) for line in output.splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', type=int, default=29635)
    args = parser.parse_args()
    source = args.source_run.resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output = args.output or source/'performance'/f'worker_sweep_{stamp}'
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'starting', 'pid': os.getpid(), 'source_run': str(source),
             'started_at': datetime.now(timezone.utc).isoformat(), 'results': [],
             'measurement': 'wall-clock elapsed between steps 50 and 190, divided by 140',
             'candidates': [2, 4, 8, 16], 'trial_steps': 200}
    child = None
    config = yaml.safe_load((source/'train_b16/launch.yaml').read_text())
    assert config['datasets']['vla_data']['per_device_batch_size'] == 16
    assert config['trainer']['gradient_accumulation_steps'] == 1
    assert config['seed'] == 42 and config['framework']['cot_training_mode'] == 'shared_canvas_v1'
    assert not config['trainer'].get('pretrained_checkpoint') and not config['trainer'].get('is_resume')
    assert len(gpu_memory()) == 8
    shutil.copyfile(__file__, output/'benchmark_script.py')
    import torch
    cuda_home = Path('/usr/local')/f'cuda-{torch.version.cuda}'
    if not (cuda_home/'bin/nvcc').is_file():
        raise FileNotFoundError(f'Missing CUDA toolkit: {cuda_home}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', WANDB_MODE='disabled',
               TOKENIZERS_PARALLELISM='false', NO_ALBUMENTATIONS_UPDATE='1', OMP_NUM_THREADS='4',
               MAX_JOBS='4', TORCHDYNAMO_DISABLE='1', CUDA_HOME=str(cuda_home),
               PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH', ''))
    env.pop('STARVLA_DISABLE_DEEPSPEED', None)

    def save(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(output/'status.json', state)

    def stop_child():
        if child is None or child.poll() is not None:
            return
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()

    def cancel(signum, _):
        raise InterruptedError(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, cancel)

    def trial(workers, repeat):
        nonlocal child
        for _ in range(12):
            if max(gpu_memory()) < 128:
                break
            time.sleep(5)
        else:
            raise RuntimeError('GPUs are still occupied; refusing to overlap trials')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port))
        run = output/f'workers_{workers}_round_{repeat}'
        run.mkdir()
        cfg = json.loads(json.dumps(config))
        cfg.update(run_root_dir=str(output), run_id=run.name, output_dir=str(run),
                   config_yaml=str(run/'launch.yaml'))
        cfg['datasets']['vla_data'].update(num_workers=workers, prefetch_factor=2, persistent_workers=True)
        cfg['trainer'].update(max_train_steps=200, save_interval=1000, logging_frequency=10)
        (run/'launch.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        shutil.copyfile(source/'train_b16/deepspeed.json', run/'deepspeed.json')
        accelerate = yaml.safe_load((source/'train_b16/accelerate.yaml').read_text())
        accelerate['deepspeed_config']['deepspeed_config_file'] = str(run/'deepspeed.json')
        (run/'accelerate.yaml').write_text(yaml.safe_dump(accelerate, sort_keys=False))
        command = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file', str(run/'accelerate.yaml'),
                   '--num_processes', '8', '--main_process_port', str(args.port),
                   str(ROOT/'starVLA/training/train_starvla.py'), '--config_yaml', str(run/'launch.yaml')]
        write_json(run/'command.json', {'command': command, 'cwd': str(ROOT)})
        print(f'Starting workers={workers}, round={repeat}', flush=True)
        save(status='benchmarking', current_workers=workers, current_round=repeat, current_run=str(run))
        resources = []
        with (run/'train.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            save(trial_pid=child.pid)
            while child.poll() is None:
                resources.append({'time': time.time(), 'load': os.getloadavg(), 'gpu_memory_mib': gpu_memory()})
                write_json(run/'resources.json', resources)
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f'Trial exited {child.returncode}: {run}')
        child = None
        metrics = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
        assert metrics[-1]['step'] == 200
        for row in metrics:
            assert all(math.isfinite(row[k]) for k in ('total_loss', 'action_dit_loss', 'cot_loss', 'grad_norm'))
            assert row['cot_samples'] > 0
        health = [json.loads(line) for line in (run/'health.jsonl').read_text().splitlines() if line.strip()]
        assert health and not any(row['stop'] for row in health)
        result = {'workers': workers, 'round': repeat, 'run': str(run),
                  'seconds_per_step': measured_seconds((run/'train.log').read_text(errors='replace')),
                  'steps': 200, 'loss_finite': True, 'health_passed': True}
        state['results'].append(result)
        write_json(run/'result.json', result)
        save()
        print('RESULT '+json.dumps(result), flush=True)
        return result

    try:
        save()
        initial = [trial(workers, 1) for workers in (2, 4, 8, 16)]
        finalists = sorted(initial, key=lambda row: row['seconds_per_step'])[:2]
        for row in finalists:
            trial(row['workers'], 2)
        selected, averages = choose_workers(state['results'])
        save(status='launching_training', selected_workers=selected, confirmed_seconds_per_step=averages)
        lines = ['# DataLoader worker sweep', '', '| Workers / GPU | Round | Seconds / step |', '|---:|---:|---:|']
        for row in state['results']:
            lines.append(f"| {row['workers']} | {row['round']} | {row['seconds_per_step']:.3f} |")
        lines += ['', f'Selected workers per GPU: **{selected}**. Differences below 5% favor fewer workers.',
                  '', 'All trials: 8 GPUs, batch 16/GPU, prefetch 2, seed 42, normal RGB Shared-CoT.',
                  'Only workers differ between candidates; all short runs start from the same pretrained model.']
        (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
        new_run = ROOT/'playground/Checkpoints'/f'groundingpipi05_eef20_h32_shared_cot_rgb_w{selected}_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}'
        launch_log = new_run.with_suffix('.launch.log')
        command = ['bash', str(HERE/'run_groundingpi_pi05_eef_lerobot_rgb_cot.sh'),
                   '--workers', str(selected), '--train-port', str(args.port)]
        with launch_log.open('w') as log:
            launched = subprocess.Popen(command, cwd=ROOT, env=dict(env, OUTPUT_PATH=str(new_run),
                                        GPUS='0,1,2,3,4,5,6,7', BATCH_SIZE='16'),
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        save(status='training_launched', new_training_run=str(new_run),
             new_training_launcher_pid=launched.pid, new_training_log=str(launch_log))
        print('TRAINING_LAUNCHED '+str(new_run), flush=True)
    except BaseException as error:
        stop_child()
        save(status='cancelled' if isinstance(error, InterruptedError) else 'failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
