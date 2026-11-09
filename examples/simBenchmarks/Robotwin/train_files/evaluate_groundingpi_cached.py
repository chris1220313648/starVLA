"""Evaluate with the verified RoboScope scene cache used by the earlier CoT run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = ROOT.parent.parent
KIT = ROOT.parent / 'RoboScope-Kit'
CACHE_SOURCE = WORKSPACE / 'experiments/roboscope-groundingpi/step50000_50x20_20261105'
SUITES = ('demo_clean', 'demo_randomized')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_cache(source, tasks, robotwin_revision):
    provenance = json.loads((source / 'provenance.json').read_text())
    if provenance['revisions']['robotwin'] != robotwin_revision:
        raise ValueError('Cached scenes were generated with a different RoboTwin revision')
    hashes = json.loads((source / 'seed-hashes.json').read_text())
    required = {f'seed-cache/{suite}/{task}.pkl' for suite in SUITES for task in tasks}
    if set(hashes) != required:
        raise ValueError('Scene cache manifest does not contain exactly the required tasks')
    for name, expected in hashes.items():
        path = source / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Missing or changed scene cache: {name}')
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--cache-source', type=Path, default=CACHE_SOURCE)
    parser.add_argument('--check-cache', action='store_true')
    parser.add_argument('--suites', nargs='+', choices=SUITES, default=list(SUITES))
    args = parser.parse_args()
    sys.path.insert(0, str(KIT / 'packages/roboscope-kit/src'))
    sys.path.insert(0, str(KIT / 'examples/groundingpi'))
    from evaluate_local import run, stop_job_children, revision, write_json
    from roboscope_kit.runtime.benchmarks.robotwin.adapter import ROBOTWIN_ALL_TASKS
    from roboscope_kit.runtime.benchmarks.robotwin.seed_cache import validate_seed_cache_dir

    gpus = args.gpus.split(',')
    if not gpus or not all(g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        parser.error('GPU IDs must be unique integers')
    robotwin = WORKSPACE / 'simulation/RoboTwin'

    def check_cache():
        hashes = verify_cache(args.cache_source, ROBOTWIN_ALL_TASKS, revision(robotwin))
        for suite in args.suites:
            validate_seed_cache_dir(args.cache_source/'seed-cache', suite, ROBOTWIN_ALL_TASKS, 20)
        return hashes

    if args.check_cache:
        print(json.dumps({'passed': True, 'cache_files': len(check_cache()),
                          'cache_source': str(args.cache_source)}))
        return
    if args.checkpoint is None or args.output is None or not args.checkpoint.is_file():
        parser.error('An existing checkpoint and a fresh output directory are required')
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'state': 'validating_cache', 'suites': {}}
    marker = str(args.output / 'parity-{pid}.json')
    parent_status = args.output.parent/'status.json'

    def save_status(state, **values):
        write_json(args.output/'summary.json', dict(summary, state=state, **values))
        if parent_status.is_file():
            status = json.loads(parent_status.read_text())
            status.update(status=state, eval_output=str(args.output), **values,
                          updated_at=datetime.now(timezone.utc).isoformat())
            write_json(parent_status, status)

    try:
        save_status('evaluating')
        hashes = check_cache()
        shutil.copytree(args.cache_source/'seed-cache', args.output/'seed-cache')
        for name, expected in hashes.items():
            if digest(args.output/name) != expected:
                raise ValueError(f'Scene cache copy differs: {name}')
        for suite in args.suites:
            validate_seed_cache_dir(args.output/'seed-cache', suite, ROBOTWIN_ALL_TASKS, 20)
        write_json(args.output/'seed-hashes.json', hashes)
        manifest = json.loads((args.cache_source/'manifest.json').read_text())
        manifest['runtime']['seed_cache_dir'] = str(args.output/'seed-cache')
        manifest['runtime']['launcher_path'] = str(KIT/'benchmarks/robotwin/roboscope_robotwin_launcher.py')
        manifest['defaults'].update(episodes=20, episode_shard_size=10, save_videos=False)
        write_json(args.output/'manifest.json', manifest)
        write_json(args.output/'policy.json', {'schema_version': 2, 'command': [
            str(WORKSPACE/'envs/starvla/bin/python'), str(KIT/'examples/groundingpi/policy.py')],
            'cwd': str(ROOT)})
        write_json(args.output/'provenance.json', {
            'purpose': 'local environment evaluation; not official benchmark scores',
            'checkpoint': str(args.checkpoint), 'checkpoint_sha256': digest(args.checkpoint),
            'seed_cache_source': str(args.cache_source/'seed-cache'),
            'episodes_per_task': 20, 'gpus': gpus, 'autotune': False,
            'revisions': {name: revision(path) for name, path in {
                'kit': KIT, 'starvla': ROOT, 'robotwin': robotwin}.items()},
            'kit_diff': subprocess.check_output(['git', '-C', str(KIT), 'diff'], text=True),
            'started': datetime.now(timezone.utc).isoformat()})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, STARVLA_ROOT=str(ROOT),
                   PYTHONPATH=os.pathsep.join([str(KIT/'packages/roboscope-kit/src'), str(ROOT)]),
                   HF_HOME='/share/project/robot-rfm/cache/huggingface', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', ROBOTWIN_SHARED_ENGINE='0',
                   GROUNDINGPI_VERIFY_PATH=marker, ROBOTWIN_INSTRUCTION_SEED_LIMIT='100')
        for suite in args.suites:
            for smoke in (True, False):
                name = f'{suite}-smoke' if smoke else suite
                summary['state'] = name
                write_json(args.output/'summary.json', summary)
                command = [sys.executable, '-m', 'roboscope_kit.runtime.benchmark',
                           '--benchmark', 'robotwin', '--suite', suite,
                           '--checkpoint-source', str(args.checkpoint),
                           '--benchmark-manifest', str(args.output/'manifest.json'),
                           '--policy-config', str(args.output/'policy.json'),
                           '--output-dir', str(args.output/name)]
                command += ['--quick-test'] if smoke else [
                    '--episodes', '20', '--concurrency', str(len(gpus)), '--replicas', str(len(gpus))]
                parity_before = set(args.output.glob('parity-*.json'))
                try:
                    run(command, args.output/f'{name}.log',
                        dict(env, CUDA_VISIBLE_DEVICES=gpus[0]) if smoke else env, ROOT, 48*3600)
                finally:
                    stop_job_children(marker)
                result = json.loads((args.output/name/'result.json').read_text())
                counts = result['result']
                errors = args.output/name/'errors.jsonl'
                parity = set(args.output.glob('parity-*.json')) - parity_before
                if (counts['state'] != 'complete' or counts['episodes'] != (1 if smoke else 1000)
                        or counts['total_steps'] <= 0 or not parity
                        or (errors.exists() and errors.stat().st_size)):
                    raise RuntimeError(f'Incomplete or errored evaluation: {name}')
                if not smoke:
                    tasks = result['official_result']['tasks']
                    if len(tasks) != 50 or any(t['episodes'] != 20 for t in tasks):
                        raise RuntimeError(f'Incomplete task coverage: {name}')
                    summary['suites'][suite] = {'result': counts, 'tasks': tasks}
                    write_json(args.output/'summary.json', summary)
        summary['state'] = 'complete'
        save_status('complete')
    except BaseException as error:
        summary.update(state='failed', error=repr(error))
        save_status('failed', error=repr(error))
        raise
    finally:
        stop_job_children(marker)
        write_json(args.output/'summary.json', summary)
        lines = ['# GroundingPI RGB RoboScope evaluation', '', f"State: {summary['state']}", '',
                 f'Checkpoint: `{args.checkpoint}`', '', f'Cached scenes: `{args.cache_source}`', '',
                 'Local protocol; not official benchmark scores.', '',
                 '| Suite | Task | Successes | Episodes | Rate |', '|---|---|---:|---:|---:|']
        for suite, value in summary['suites'].items():
            for task in value['tasks'] + [dict(value['result'], task='**Total**')]:
                lines.append(f"| {suite} | {task['task']} | {task['successes']} | "
                             f"{task['episodes']} | {task['success_rate']:.1%} |")
        (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
