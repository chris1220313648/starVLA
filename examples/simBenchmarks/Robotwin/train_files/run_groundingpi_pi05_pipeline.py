"""Validate, select a fitting batch, train GroundingPIPI05, verify, and evaluate.

The supervisor survives terminal disconnects when started with nohup. status.json
records its current stage. Each OOM attempt uses a fresh directory and fresh
weights. EEF mode increases accumulation when shrinking the microbatch to keep
the initial global batch fixed; the original joint mode keeps accumulation one.
Other failures stop the pipeline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
EVAL = HERE.parent / 'eval_files'
OOM_MARKERS = ('CUDA out of memory', 'torch.OutOfMemoryError', 'CUDA error: out of memory')


def is_oom(log):
    return any(marker in log for marker in OOM_MARKERS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh experiment directory')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, choices=(1,2,4,8,16), default=16)
    parser.add_argument('--config-yaml', type=Path)
    parser.add_argument('--tune-batch', action='store_true', help='Benchmark all fitting microbatches at fixed effective batch')
    parser.add_argument('--strict-batch', action='store_true', help='Stop on OOM instead of reducing the microbatch')
    parser.add_argument('--knowledge-insulation', action='store_true')
    parser.add_argument('--color-jitter', type=float, default=None)
    parser.add_argument('--filter-static-segments', action='store_true')
    parser.add_argument('--eef', action='store_true', help='EEF20, continuous expert state, 32 actions, 384x320 composed image')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--with-cot', action='store_true')
    parser.add_argument('--image-input-mode', choices=('composite', 'separate'), default='composite')
    parser.add_argument('--train-only', action='store_true', help='Verify the final checkpoint and stop without evaluation')
    parser.add_argument('--cot-root', type=Path, default=Path(
        '/share/project/robot-rfm/robot-processed-data/lerobot_v3_data/robotwin2.0_Lerobotv3_cot/clean/20261029_v1'))
    parser.add_argument('--cot-loss-weight', type=float, default=1.0)
    parser.add_argument('--instruction-source', choices=('default', 'seen'), default='default')
    parser.add_argument('--raw-data', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--smoke-steps', type=int, default=3)
    parser.add_argument('--train-port', type=int, default=29615)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.color_jitter is None:
        from omegaconf import OmegaConf
        args.color_jitter = float(OmegaConf.load(args.config_yaml).datasets.vla_data.get('color_jitter', 0.0)) if args.config_yaml else 0.0
    if args.tune_batch and (not args.eef or args.smoke_steps < 20):
        parser.error('--tune-batch requires --eef and --smoke-steps >=20')
    import math
    if args.dataset_root is not None and not args.eef:
        parser.error('--dataset-root requires --eef')
    if args.with_cot and args.dataset_root is None:
        parser.error('--with-cot requires --dataset-root')
    if args.instruction_source == 'seen' and args.dataset_root is None:
        parser.error('--instruction-source seen requires --dataset-root')
    if args.image_input_mode == 'separate' and (not args.eef or args.dataset_root is None):
        parser.error('separate images require the pi05_eef LeRobot dataset')
    if not math.isfinite(args.cot_loss_weight) or args.cot_loss_weight <= 0:
        parser.error('--cot-loss-weight must be finite and positive')
    if not 0 <= args.color_jitter <= 1:
        parser.error('color-jitter must be in [0, 1]')
    if (args.color_jitter or args.filter_static_segments) and not args.eef:
        parser.error('Color jitter and static resampling require --eef')
    if args.eef and args.knowledge_insulation:
        parser.error('Continuous expert state does not support knowledge insulation')
    gpus = args.gpus.split(',')
    if len(gpus) != len(set(gpus)) or not all(g.isdigit() for g in gpus):
        parser.error('GPU IDs must be unique integers')
    if args.steps < 1 or args.smoke_steps < 3 or args.workers < 0:
        parser.error('steps must be positive, smoke-steps >=3 and workers >=0')
    if args.save_interval < 1:
        parser.error('--save-interval must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'launch_args.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    for command, filename in ((['git', 'rev-parse', 'HEAD'], 'git_revision.txt'),
                              (['git', 'diff', '--binary'], 'source_changes.patch'),
                              ([sys.executable, '-m', 'pip', 'freeze'], 'environment.txt')):
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        (output / filename).write_text(result.stdout)
    with tarfile.open(output / 'source_snapshot.tar.gz', 'w:gz') as archive:
        for name in ('starVLA', 'examples/simBenchmarks/Robotwin', 'deployment', 'tests'):
            archive.add(ROOT / name, arcname=name,
                        filter=lambda item: None if '__pycache__' in Path(item.name).parts else item)
    image_size = ([240,320] if args.image_input_mode == 'separate' else
                  [384,320] if args.eef else [224,224])
    state = dict(status='starting', pid=os.getpid(), gpus=gpus, steps=args.steps,
                 batch_size=args.batch_size, image_size=image_size, image_input_mode=args.image_input_mode,
                 with_cot=args.with_cot, eef=args.eef, knowledge_insulation=args.knowledge_insulation, attempts=[])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus,
               NO_ALBUMENTATIONS_UPDATE='1', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1',
               # Preserve FP32 expert parameters at serving; predict_action autocasts.
               ROBOTWIN_USE_BF16='0', ROBOTWIN_POLICY_SEED='42')
    child = None

    def save(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        temp = output / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(output / 'status.json')
        print(json.dumps(values), flush=True)

    def stop_child():
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    def cancel(signum, _):
        raise InterruptedError(signum)

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)

    def run(command, log):
        nonlocal child
        with log.open('w') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        # Accelerate should reap all ranks. Also terminate any leftover group
        # before retrying an OOM, so ranks cannot hold GPU memory.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        child = None
        return code

    def training_command(dest, batch, smoke=False):
        extra = []
        if args.config_yaml:
            extra += ['--config-yaml', str(args.config_yaml.resolve())]
        extra += ['--color-jitter', str(args.color_jitter)]
        if args.filter_static_segments:
            extra.append('--filter-static-segments')
        if args.dataset_root is not None:
            extra += ['--dataset-root', str(args.dataset_root), '--cot-root', str(args.cot_root),
                      '--cot-loss-weight', str(args.cot_loss_weight), '--image-input-mode', args.image_input_mode,
                      '--instruction-source', args.instruction_source]
            if args.with_cot:
                extra.append('--with-cot')
        for name in ('raw_data', 'data_root'):
            value = getattr(args, name)
            if value is not None:
                extra += ['--' + name.replace('_', '-'), str(value)]
        return [sys.executable, str(HERE / 'groundingpi_robotwin.py'), 'train',
                '--framework', 'pi05_eef' if args.eef else 'pi05', '--image-size', '224', '--gpus', str(len(gpus)),
                '--image-input-mode', args.image_input_mode,
                '--batch-size', str(batch), '--gradient-accumulation-steps', str(args.batch_size // batch if args.eef else 1),
                '--steps', str(args.smoke_steps if smoke else args.steps),
                '--workers', str(args.workers), '--train-port', str(args.train_port),
                '--num-warmup-steps', str(1 if smoke else min(500, max(1,args.steps//10))),
                '--health-check-interval', str(1 if smoke else 100),
                '--save-interval', str(args.smoke_steps if smoke else args.save_interval),
                '--output', str(dest)] + extra + (['--knowledge-insulation'] if args.knowledge_insulation else [])

    try:
        save(status='validating')
        validation = training_command(output, args.batch_size)
        validation[2] = 'validate'
        code = run(validation, output / 'validation.log')
        if code:
            raise RuntimeError('Data validation failed; see validation.log')
        batch = args.batch_size
        if args.tune_batch:
            import statistics
            candidates = []
            probe = batch
            while probe:
                save(status='benchmarking', batch_size=probe)
                with tempfile.TemporaryDirectory(prefix='robotwin_pi05_benchmark_', dir=output) as temp:
                    smoke = Path(temp) / 'run'
                    log = output / f'benchmark_b{probe}.log'
                    code = run(training_command(smoke, probe, smoke=True), log)
                    detail = (smoke / 'train.log').read_text() if (smoke / 'train.log').exists() else ''
                    (output / f'benchmark_b{probe}_train.log').write_text(detail)
                    if code:
                        state['attempts'].append(dict(stage='benchmark', batch_size=probe, exit_code=code))
                        if not is_oom(detail + log.read_text()):
                            raise RuntimeError(f'Batch benchmark failed at {probe}; see logs')
                    else:
                        metrics = [json.loads(line) for line in (smoke / 'metrics.jsonl').read_text().splitlines()]
                        for filename in ('metrics.jsonl', 'health.jsonl'):
                            shutil.copy2(smoke / filename, output / f'benchmark_b{probe}_{filename}')
                        measured = [row for row in metrics if row['step'] > 5]
                        if len(measured) < 15:
                            raise RuntimeError('Benchmark requires at least 15 measured optimizer steps')
                        record = dict(batch_size=probe,
                                      samples_per_second=args.batch_size * len(gpus) / statistics.median(row['timing/optimizer_step'] for row in measured),
                                      peak_memory_fraction=max(row['memory/peak_fraction'] for row in metrics))
                        state['attempts'].append(dict(stage='benchmark', exit_code=0, **record))
                        if record['peak_memory_fraction'] <= .9:
                            candidates.append(record)
                    save()
                probe //= 2
            if not candidates:
                raise RuntimeError('No batch passed with 10% memory headroom')
            best_speed = max(row['samples_per_second'] for row in candidates)
            chosen = min((row for row in candidates if row['samples_per_second'] >= best_speed * .97),
                         key=lambda row: row['peak_memory_fraction'])
            batch = chosen['batch_size']
            save(status='batch_selected', selected_batch=chosen)
        while batch:
            save(status='smoke_training', batch_size=batch)
            with tempfile.TemporaryDirectory(prefix='robotwin_pi05_smoke_', dir=output) as temp:
                smoke = Path(temp) / 'run'
                log = output / f'smoke_b{batch}.log'
                code = run(training_command(smoke, batch, smoke=True), log)
                detail = (smoke / 'train.log').read_text() if (smoke / 'train.log').exists() else ''
                (output / f'smoke_b{batch}_train.log').write_text(detail)
                state['attempts'].append(dict(stage='smoke', batch_size=batch, exit_code=code))
                if code:
                    if is_oom(detail + log.read_text()) and not args.strict_batch:
                        batch //= 2
                        continue
                    raise RuntimeError(f'Smoke training failed at batch {batch}')
                save(status='verifying_smoke')
                code = run([sys.executable, str(HERE / 'verify_groundingpi_pi05.py'),
                            '--run', str(smoke)], output / 'smoke_verification.log')
                if code:
                    raise RuntimeError('Smoke checkpoint verification failed')
                for name in ('verification.json', 'health.jsonl', 'metrics.jsonl'):
                    shutil.copy2(smoke / name, output / ('smoke_' + name))
            run_dir = output / f'train_b{batch}'
            save(status='training', run_dir=str(run_dir), batch_size=batch,
                 global_batch_size=(args.batch_size if args.eef else batch)*len(gpus),
                 gradient_accumulation_steps=(args.batch_size // batch if args.eef else 1))
            log = output / f'train_b{batch}_launch.log'
            code = run(training_command(run_dir, batch), log)
            state['attempts'].append(dict(stage='train', batch_size=batch, exit_code=code))
            if code:
                detail = (run_dir / 'train.log').read_text() if (run_dir / 'train.log').exists() else ''
                if is_oom(detail + log.read_text()) and not args.strict_batch:
                    batch //= 2
                    continue
                raise RuntimeError('Formal training failed')
            break
        if not batch:
            raise RuntimeError('CUDA OOM even with batch 1')
        save(status='verifying_final')
        code = run([sys.executable, str(HERE / 'verify_groundingpi_pi05.py'),
                    '--run', str(run_dir)], output / 'final_verification.log')
        if code:
            raise RuntimeError('Final checkpoint verification failed')
        if args.train_only:
            save(status='complete')
            return
        save(status='evaluating')
        command = [sys.executable, str(EVAL / 'schedule_groundingpi_eval.py'),
                   '--run-dir', str(run_dir), '--checkpoint', str(run_dir / 'final_model/pytorch_model.pt'),
                   '--delay-hours', '0', '--retry-failed', '2', '--episodes', '20', '--gpus', args.gpus,
                   '--poll-seconds', '10', '--deploy-template', str(EVAL / ('deploy_groundingpi_pi05_eef.yml' if args.eef else 'deploy_groundingpi_pi05.yml')),
                   '--output', str(output / 'eval')]
        if run(command, output / 'evaluation.log'):
            raise RuntimeError('Evaluation incomplete; see eval/status.json and task logs')
        save(status='complete')
    except BaseException as exc:
        stop_child()
        save(status='cancelled' if isinstance(exc, InterruptedError) else 'failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
