"""OAT-specific checks for the shared eight-worker LIBERO evaluator."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from collections import Counter


def aggregate(output, task_start, task_end, trials, workers, failure_code=0):
    rows = [json.loads(line) for p in output.glob('worker_*/episodes.jsonl')
            for line in p.read_text().splitlines()]
    expected = {(task, trial) for task in range(task_start, task_end + 1) for trial in range(trials)}
    actual = [(r['task'], r['trial']) for r in rows]
    reports = [json.loads(p.read_text()) for p in output.glob('worker_*/results.json')]
    errors = sum(r['error'] is not None for r in rows)
    invalid = sum(r['invalid_output'] for r in rows)
    complete = (not failure_code and len(actual) == len(expected) and set(actual) == expected and
                len(reports) == min(workers, len(expected)) and all(r['complete'] for r in reports))
    result = dict(complete=complete, valid=complete and not errors and not invalid,
                  expected=len(expected), completed=len(rows),
                  successes=sum(r['success'] and not r['error'] for r in rows), errors=errors,
                  invalid_outputs=invalid, failure_code=failure_code,
                  missing=sorted(expected - set(actual)), duplicate_count=len(actual) - len(set(actual)),
                  per_task={str(t): dict(expected=trials,
                      completed=sum(r['task'] == t for r in rows),
                      successes=sum(r['task'] == t and r['success'] and not r['error'] for r in rows))
                      for t in range(task_start, task_end + 1)})
    result['success_rate'] = result['successes'] / len(expected) if result['valid'] else None
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result['valid']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare', 'health', 'aggregate'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--smoke-only', type=int, default=0)
    p.add_argument('--base-port', type=int, default=6800)
    p.add_argument('--pid', type=int)
    p.add_argument('--task-start', type=int, default=0)
    p.add_argument('--task-end', type=int, default=9)
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--failure-code', type=int, default=0)
    a = p.parse_args()

    if a.mode == 'aggregate':
        raise SystemExit(0 if aggregate(a.output, a.task_start, a.task_end, a.trials, a.workers, a.failure_code) else 1)
    if a.mode == 'health':
        import websockets.sync.client
        from deployment.model_server.tools import msgpack_numpy
        for key in list(os.environ):
            if key.lower().endswith('_proxy'):
                os.environ.pop(key)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            os.kill(a.pid, 0)
            try:
                with websockets.sync.client.connect(f'ws://127.0.0.1:{a.base_port}',
                        compression=None, max_size=None, open_timeout=2) as conn:
                    meta = msgpack_numpy.unpackb(conn.recv(timeout=5))
            except (OSError, TimeoutError):
                time.sleep(1)
                continue
            if (Path(meta['ckpt_path']).resolve() != a.checkpoint.resolve() or
                    meta.get('action_codec') != 'oat' or meta.get('action_chunk_size') != 32 or
                    'franka' not in meta.get('available_unnorm_keys', [])):
                raise ValueError(f'Policy handshake does not match OAT checkpoint: {meta}')
            print(json.dumps(meta), flush=True)
            return
        raise TimeoutError('OAT policy startup timed out')

    if not 0 <= a.task_start <= a.task_end <= 9 or not 1 <= a.trials <= 50:
        raise ValueError('LIBERO-10 requires tasks 0..9 and 1..50 trials per task')
    gpus = a.gpus.split(',')
    if len(gpus) != a.workers or not 1 <= a.workers <= 8 or len(set(gpus)) != len(gpus):
        raise ValueError('GPU list must contain 1..8 distinct IDs')
    if not a.smoke_only and a.workers != 8:
        raise ValueError('Formal evaluation requires eight GPUs; use --smoke-only for a smaller smoke')
    root = a.checkpoint.resolve().parent.parent
    import yaml
    config = yaml.safe_load((root / 'config.yaml').read_text())
    framework = config['framework']
    model = framework['u0']
    action = framework['action_model']
    if framework['name'].lower() != 'u0oat' or (model['sequence_h'], action['action_horizon'], action['action_dim']) != (2, 32, 7):
        raise ValueError('Checkpoint config is not the expected U0OAT h=2 [32,7] policy')
    if not a.checkpoint.is_file():
        raise FileNotFoundError(a.checkpoint)
    for gpu in gpus:
        occupied = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if occupied:
            raise RuntimeError(f'GPU {gpu} is occupied; refusing to interfere')
    for port in range(a.base_port, a.base_port + a.workers):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(checkpoint=str(a.checkpoint.resolve()), gpus=gpus, suite='libero_10',
                  task_start=a.task_start, task_end=a.task_end, trials_per_task=a.trials,
                  workers=a.workers, smoke_only=bool(a.smoke_only),
                  seed=7, action_codec='oat', action_horizon=32,
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    (a.output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PREFLIGHT_OK {a.output}', flush=True)


if __name__ == '__main__':
    main()
