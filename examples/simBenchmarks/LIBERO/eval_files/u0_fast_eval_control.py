"""Preflight, websocket readiness, and exact coverage checks for U0Fast evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def aggregate(output, task_start, task_end, trials, failure_code=0):
    rows = [json.loads(line) for p in output.glob('worker_*/episodes.jsonl')
            for line in p.read_text().splitlines()]
    expected = {(task, trial) for task in range(task_start, task_end+1) for trial in range(trials)}
    actual = [(r['task'], r['trial']) for r in rows]
    reports = [json.loads(p.read_text()) for p in output.glob('worker_*/results.json')]
    errors = sum(r['error'] is not None for r in rows)
    invalid = sum(r['invalid_output'] for r in rows)
    complete = (not failure_code and len(actual) == len(expected) and set(actual) == expected and
                len(reports) == min(8, len(expected)) and all(r['complete'] for r in reports))
    valid = complete and not errors and not invalid
    result = dict(complete=complete, valid=valid, expected=len(expected), completed=len(rows),
                  successes=sum(r['success'] and not r['error'] for r in rows), errors=errors,
                  invalid_outputs=invalid, failure_code=failure_code,
                  missing=sorted(expected-set(actual)), duplicate_count=len(actual)-len(set(actual)),
                  per_task={str(t): dict(expected=trials,
                      completed=sum(r['task']==t for r in rows),
                      successes=sum(r['task']==t and r['success'] and not r['error'] for r in rows))
                      for t in range(task_start, task_end+1)})
    result['success_rate'] = result['successes']/len(expected) if valid else None
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return valid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare','health','aggregate'])
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--base-port', type=int, default=6800)
    p.add_argument('--pid', type=int)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--task-start', type=int, default=0)
    p.add_argument('--task-end', type=int, default=9)
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--failure-code', type=int, default=0)
    a = p.parse_args()
    if a.mode == 'aggregate':
        raise SystemExit(0 if aggregate(a.output,a.task_start,a.task_end,a.trials,a.failure_code) else 1)
    if a.mode == 'health':
        import websockets.sync.client
        from deployment.model_server.tools import msgpack_numpy
        for key in list(os.environ):
            if key.lower().endswith('_proxy'):
                os.environ.pop(key)
        deadline = time.monotonic()+a.timeout
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
                    meta['action_chunk_size'] != 8 or 'franka' not in meta['available_unnorm_keys']):
                raise ValueError('Policy handshake does not match requested checkpoint/normalization/horizon')
            print(json.dumps(meta), flush=True)
            return
        raise TimeoutError('Policy startup timed out')
    import yaml
    import torch
    from safetensors import safe_open
    if not 0 <= a.task_start <= a.task_end <= 9 or not 1 <= a.trials <= 50:
        raise ValueError('Goal requires tasks 0..9 and trials 1..50')
    gpu_ids = a.gpus.split(',')
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8:
        raise ValueError('Expected eight distinct GPU IDs')
    for gpu in gpu_ids:
        occupied = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:
            raise RuntimeError(f'GPU {gpu} is occupied; refusing to interfere')
    for port in range(a.base_port,a.base_port+8):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',port))
    root = a.checkpoint.resolve().parent.parent
    config = yaml.safe_load((root/'config.yaml').read_text())
    stats = json.loads((root/'dataset_statistics.json').read_text())
    framework = config['framework']
    if framework['name'] != 'U0Fast' or (framework['action_model']['action_horizon'],framework['action_model']['action_dim']) != (8,7):
        raise ValueError('Expected U0Fast [8,7] checkpoint')
    assert len(stats['franka']['action']['q01']) == 7
    with safe_open(a.checkpoint, framework='pt', device='cpu') as f:
        if not torch.equal(f.get_tensor('action_token_ids'),torch.arange(149595,151643)):
            raise ValueError('Wrong FAST token mapping')
        if len(f.keys()) != 400 or any(k != 'action_token_ids' and not k.startswith('u0_interface.model.') for k in f.keys()):
            raise ValueError('Incomplete U0 model tensor inventory')
    assets = [a.checkpoint,root/'config.yaml',root/'dataset_statistics.json']
    if int(framework['u0'].get('sequence_h', 1)) > 1:
        assets.append(Path('starVLA/model/modules/vlm/u0_sequence.py'))
    assets.extend(Path(p) for p in (
        __file__, 'examples/simBenchmarks/LIBERO/eval_files/eval_libero.py',
        'examples/simBenchmarks/LIBERO/eval_files/model2libero_interface.py',
        'examples/simBenchmarks/LIBERO/eval_files/run_u0_fast_libero_goal_8gpu.sh',
        'starVLA/model/framework/VLM4A/U0Fast.py', 'starVLA/model/modules/vlm/U0.py',
        'starVLA/model/modules/vlm/Emu3_5.py', 'deployment/model_server/policy_wrapper.py'))
    for base in (framework['u0']['base_vlm'], framework['u0']['vision_tokenizer'],framework['action_model']['fast_tokenizer_name']):
        folder = Path(base)
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        assets.extend(p for p in folder.rglob('*') if p.is_file() and '.cache' not in p.parts)
    report = dict(checkpoint=str(a.checkpoint.resolve()),gpus=gpu_ids,task_start=a.task_start,
                  task_end=a.task_end,trials=a.trials,seed=7,action_horizon=8,unnorm_key='franka',
                  git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  hashes={str(p.resolve()):sha256(p) for p in assets})
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PREFLIGHT_OK {a.output}',flush=True)


if __name__ == '__main__':
    main()
