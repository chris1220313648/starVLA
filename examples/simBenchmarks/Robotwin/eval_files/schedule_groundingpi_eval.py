#!/usr/bin/env python3
"""Delay, wait for idle GPUs, then run all clean tasks followed by randomized tasks.

Run with --dry-run to inspect commands without creating files or using GPUs.
The scheduler can be detached with nohup; SIGTERM cancels its active eval group.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = Path('/share/project/robot-rfm/robot-exp/usr/cjy')
DEFAULT_RUN = ROOT / 'playground/Checkpoints/groundingpipi_v3_robotwin_clean_448_8gpu_b4_a4_20261024_102106'
STAGES = [('clean2clean', 'demo_clean'), ('clean2rand', 'demo_randomized')]


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'[{now()}] {message}', flush=True)


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def latest_checkpoint(run_dir):
    """A summary row is appended only AFTER torch.save returns successfully."""
    candidates = []
    for line in (run_dir / 'summary.jsonl').read_text().splitlines():
        try:
            step = int(json.loads(line)['steps'])
        except (ValueError, KeyError, TypeError):
            continue
        path = run_dir / 'checkpoints' / f'steps_{step}_pytorch_model.pt'
        if path.is_file() and path.stat().st_size > 0:
            candidates.append((step, path))
    if not candidates:
        raise RuntimeError(f'No completed checkpoint in {run_dir}')
    return max(candidates, key=lambda item: item[0])


def task_names():
    text = (HERE / 'start_eval.sh').read_text()
    block = re.search(r'ROBOTWIN_ALL_TASKS=\((.*?)\)', text, re.S).group(1)
    return block.split()


def gpu_memory(gpus):
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'
    ], text=True)
    usage = {int(row.split(',')[0]): int(row.split(',')[1]) for row in output.splitlines()}
    return {gpu: usage[gpu] for gpu in gpus}


def training_running(run_dir):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (OSError, ProcessLookupError):
            continue
        if 'train_starvla.py' in args and str(run_dir) in args:
            return True
    return False


def stage_command(mode, checkpoint, name):
    return ['bash', str(HERE / 'start_eval.sh'), '-m', mode, '-n', name,
            '-c', str(checkpoint), '-s', '0', '-j', '1', '-p', '15694', 'all']


def summarize(stage_dir, tasks, episodes, robotwin, mode, name, result_root=None):
    result_root = Path(result_root) if result_root is not None else robotwin / 'eval_result'
    results = []
    for task in tasks:
        files = list((result_root / task / 'model2robotwin_interface' / mode / name).glob('*/_result.txt'))
        rate = None
        if files:
            try:
                rate = float(max(files, key=lambda p: p.stat().st_mtime).read_text().strip().splitlines()[-1])
                if not 0 <= rate <= 1:
                    rate = None
            except (ValueError, OSError):
                pass
        results.append({'task': task, 'success_rate': rate,
                        'successes': round(rate * episodes) if rate is not None else None,
                        'episodes': episodes if rate is not None else 0,
                        'result_file': str(files[0]) if len(files) == 1 else None})
    complete = [row for row in results if row['success_rate'] is not None]
    summary = {'completed_tasks': len(complete), 'total_tasks': len(tasks),
               'failed_tasks': [row['task'] for row in results if row['success_rate'] is None],
               'mean_success_rate_completed': (sum(row['success_rate'] for row in complete) / len(complete)) if complete else None,
               'successes': sum(row['successes'] for row in complete),
               'episodes': sum(row['episodes'] for row in complete), 'tasks': results}
    write_json(stage_dir / 'summary.json', summary)
    return summary


def write_report(output, state):
    lines = ['# RoboTwin evaluation', '',
             f"Checkpoint: `{state.get('checkpoint', 'pending')}`", '',
             f"Status: {state['status']}; seed=0; instruction_type=unseen; "
             f"episodes per task={state['episodes_per_task']}", '',
             '| Environment | Tasks complete | Successes / episodes | Success rate (completed tasks) |',
             '| --- | --- | --- | --- |']
    for label, _ in STAGES:
        summary = state['stages'].get(label, {}).get('summary')
        if summary is None:
            continue
        rate = summary['mean_success_rate_completed']
        display = f'{rate:.2%}' if rate is not None else 'N/A'
        lines.append(f"| {label} | {summary['completed_tasks']}/50 | "
                     f"{summary['successes']}/{summary['episodes']} | {display} |")
    for label, stage in state['stages'].items():
        summary = stage.get('summary')
        if summary is None:
            continue
        lines += ['', f'## {label}', '', '| Task | Successes / episodes | Success rate |',
                  '| --- | --- | --- |']
        for row in summary['tasks']:
            rate = row['success_rate']
            display = f'{rate:.0%}' if rate is not None else 'MISSING (infrastructure failure)'
            lines.append(f"| {row['task']} | {row['successes']}/{row['episodes']} | {display} |")
    (output / 'evaluation_summary.md').write_text('\n'.join(lines) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--delay-hours', type=float, default=13)
    parser.add_argument('--retry-failed', type=int, default=0, help='Retries for tasks missing result files')
    parser.add_argument('--checkpoint', type=Path, help='Evaluate this completed checkpoint instead of selecting latest')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--idle-memory-mib', type=int, default=4096)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deploy-template', type=Path, default=HERE / 'deploy_groundingpi_v3.yml')
    parser.add_argument('--starvla-python', type=Path, default=BASE / 'envs/starvla/bin/python')
    parser.add_argument('--robotwin-python', type=Path, default=BASE / 'envs/robotwin/bin/python')
    parser.add_argument('--robotwin-path', type=Path, default=BASE / 'simulation/RoboTwin')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.delay_hours < 0 or args.episodes <= 0 or args.poll_seconds <= 0 or args.retry_failed < 0:
        parser.error('delay-hours and retry-failed must be >= 0; episodes and poll-seconds must be > 0')
    args.run_dir = args.run_dir.resolve()
    return args


def main():
    args = parse_args()
    args.checkpoint = getattr(args, 'checkpoint', None)
    args.retry_failed = getattr(args, 'retry_failed', 0)
    gpus = [int(value) for value in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('GPU list must be nonempty and unique')
    for path in [args.starvla_python, args.robotwin_python]:
        if not os.access(path, os.X_OK):
            raise FileNotFoundError(f'Python not executable: {path}')
    for path in [args.run_dir / 'config.yaml', args.run_dir / 'dataset_statistics.json',
                 args.run_dir / 'summary.jsonl', args.robotwin_path / 'script/eval_policy.py',
                 getattr(args, 'deploy_template', HERE / 'deploy_groundingpi_v3.yml')]:
        if not path.is_file():
            raise FileNotFoundError(path)
    tasks = task_names()
    if len(tasks) != 50:
        raise RuntimeError(f'Expected 50 tasks, found {len(tasks)}')
    # Check GPU IDs now; no CUDA context or model is created.
    gpu_memory(gpus)
    due = datetime.now(timezone.utc) + timedelta(hours=args.delay_hours)
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(checkpoint)
        if checkpoint.parent.parent != args.run_dir:
            raise ValueError('Checkpoint must belong to run-dir')
        cfg = yaml.safe_load((args.run_dir / 'config.full.yaml').read_text())
        if checkpoint.parent.name == 'final_model':
            step = int(cfg['trainer']['max_train_steps'])
        else:
            match = re.fullmatch(r'steps_(\d+)_pytorch_model.pt', checkpoint.name)
            if checkpoint.parent.name != 'checkpoints' or match is None:
                raise ValueError('Expected a final_model or steps_N_pytorch_model.pt checkpoint')
            step = int(match.group(1))
    else:
        step, checkpoint = latest_checkpoint(args.run_dir)
    output = (args.output or args.run_dir / 'eval' /
              f'step{step}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}').resolve()
    result_root = output / 'results'
    if args.dry_run:
        print(json.dumps({'due_utc': due.isoformat(), 'output': str(output),
                          'gpus': gpus, 'episodes_per_task': args.episodes,
                          'checkpoint_now': str(checkpoint), 'checkpoint_policy': 'explicit' if args.checkpoint else 'latest completed at actual eval start',
                          'starvla_python': str(args.starvla_python), 'robotwin_python': str(args.robotwin_python),
                          'robotwin_path': str(args.robotwin_path),
                          'result_root': str(result_root),
                          'deploy_template': str(args.deploy_template.resolve()),
                          'commands': [shlex.join(stage_command(mode, checkpoint, label)) for label, mode in STAGES]}, indent=2))
        return 0

    # Prevent two schedulers from evaluating the same run simultaneously.
    lock = (args.run_dir / '.robotwin_eval.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A launch bundle may already exist, but a previously started run is immutable.
    if (output / 'status.json').exists():
        raise FileExistsError(f'Evaluation already started in {output}; use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    template = getattr(args, 'deploy_template', HERE / 'deploy_groundingpi_v3.yml').resolve()
    saved_template = output / 'deploy_policy.yml'
    if template != saved_template:
        shutil.copy2(template, saved_template)
    (output / 'scheduler.pid').write_text(f'{os.getpid()}\n')
    state = {'status': 'scheduled', 'pid': os.getpid(), 'scheduled_at': now(),
             'due_utc': due.isoformat(), 'run_dir': str(args.run_dir), 'gpus': gpus,
             'episodes_per_task': args.episodes, 'tasks_per_stage': len(tasks), 'stages': {}}
    state['deploy_template'] = str(saved_template)
    state['result_root'] = str(result_root)
    child = None

    class EvaluationCancelled(Exception):
        pass

    def cancel(signum, _frame):
        # Unwind Popen.wait before waiting again: signal handlers run on the
        # same thread and can otherwise deadlock on its non-reentrant wait lock.
        raise EvaluationCancelled(signum)

    def stop_active(signum):
        nonlocal child
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state.update(status='cancelled', finished_at=now())
        write_json(output / 'status.json', state)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    write_json(output / 'status.json', state)
    log(f'Scheduled for {due.isoformat()}; output={output}; pid={os.getpid()}')
    try:
        while datetime.now(timezone.utc) < due:
            time.sleep(min(60, max(0, (due - datetime.now(timezone.utc)).total_seconds())))
        state['status'] = 'waiting_for_gpus'
        write_json(output / 'status.json', state)
        # Two consecutive idle checks avoid launching during a transient lull.
        idle_checks = 0
        while idle_checks < 2:
            usage = gpu_memory(gpus)
            training = training_running(args.run_dir)
            idle_checks = idle_checks + 1 if not training and max(usage.values()) < args.idle_memory_mib else 0
            log(f'GPU check: training_running={training}, used_MiB={usage}, idle_checks={idle_checks}/2')
            if idle_checks < 2:
                time.sleep(args.poll_seconds)
        if args.checkpoint is None:
            step, checkpoint = latest_checkpoint(args.run_dir)
        state.update(status='running', started_at=now(), checkpoint=str(checkpoint), checkpoint_step=step)
        write_json(output / 'status.json', state)
        write_json(output / 'manifest.json', {
            'checkpoint': str(checkpoint), 'checkpoint_step': step,
            'run_dir': str(args.run_dir), 'output': str(output),
            'result_root': str(result_root), 'deploy_template': str(saved_template),
            'episodes_per_task': args.episodes, 'tasks_per_stage': tasks,
            'seed': 0, 'policy_seed': os.environ.get('ROBOTWIN_POLICY_SEED'),
            'instruction_type': 'unseen', 'gpus': gpus, 'started_at': now()})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus,
                   STARVLA_PYTHON=str(args.starvla_python), ROBOTWIN_PYTHON=str(args.robotwin_python),
                   ROBOTWIN_PATH=str(args.robotwin_path), ROBOTWIN_TEST_NUM=str(args.episodes),
                   DEPLOY_POLICY_TEMPLATE_PATH=str(saved_template),
                   ROBOTWIN_EVAL_RESULT_ROOT=str(result_root),
                   ROBOTWIN_POLICY_NAME='model2robotwin_interface',
                   ROBOTWIN_AUTO_INSTALL_DEPS='0', PYTHONUNBUFFERED='1',
                   PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        failed = False
        for label, mode in STAGES:
            stage_dir = output / label
            stage_dir.mkdir()
            name = f'{args.run_dir.name}_{output.name}_{label}_step{step}'
            env['ROBOTWIN_LOG_ROOT'] = str(stage_dir)
            command = stage_command(mode, checkpoint, name)
            state['stages'][label] = {'status': 'running', 'started_at': now(), 'command': command}
            write_json(output / 'status.json', state)
            log(f'Starting {label}: {shlex.join(command)}')
            with (stage_dir / 'launch.log').open('w') as stream:
                child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                code = child.wait()
                child = None
            summary = summarize(stage_dir, tasks, args.episodes, args.robotwin_path, mode, name,
                                result_root=result_root)
            for attempt in range(args.retry_failed):
                if not summary['failed_tasks']:
                    break
                retry_command = command[:-1] + summary['failed_tasks']
                log(f'Retrying infrastructure failures: {summary["failed_tasks"]}')
                with (stage_dir / f'retry_{attempt+1}.log').open('w') as stream:
                    child = subprocess.Popen(retry_command, env=env, cwd=ROOT, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    code = child.wait()
                    child = None
                summary = summarize(stage_dir, tasks, args.episodes, args.robotwin_path, mode, name,
                                    result_root=result_root)
            stage_failed = code != 0 or summary['completed_tasks'] != len(tasks)
            failed |= stage_failed
            state['stages'][label].update(status='failed' if stage_failed else 'complete',
                                         exit_code=code, finished_at=now(), summary=summary)
            write_json(output / 'status.json', state)
            write_report(output, state)
            log(f'{label} finished: exit_code={code}, completed_tasks={summary["completed_tasks"]}/50')
        state.update(status='complete_with_failures' if failed else 'complete', finished_at=now())
        write_json(output / 'status.json', state)
        write_report(output, state)
        return int(failed)
    except EvaluationCancelled as exc:
        stop_active(exc.args[0])
    except Exception as exc:
        state.update(status='failed', error=str(exc), finished_at=now())
        write_json(output / 'status.json', state)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
