"""Regression checks for checkpoint selection, sequential eval, and observations."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'examples/simBenchmarks/Robotwin/eval_files'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, EVAL / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latest_checkpoint_requires_completed_summary(tmp_path):
    scheduler = load('robotwin_scheduler', 'schedule_groundingpi_eval.py')
    (tmp_path / 'checkpoints').mkdir()
    for step in [1000, 2000, 3000]:
        (tmp_path / f'checkpoints/steps_{step}_pytorch_model.pt').write_bytes(b'checkpoint')
    # The higher file exists but its save has not finished. Ignore an incomplete row too.
    (tmp_path / 'summary.jsonl').write_text('{"steps":1000}\n{"steps":2000}\n{"steps":')
    assert scheduler.latest_checkpoint(tmp_path)[0] == 2000
    (tmp_path / 'checkpoints/steps_2000_pytorch_model.pt').unlink()
    assert scheduler.latest_checkpoint(tmp_path)[0] == 1000


@pytest.mark.parametrize('cancelled', [False, True])
@pytest.mark.parametrize('explicit', [False, True])
def test_clean_failure_still_runs_rand_with_same_weight(tmp_path, monkeypatch, cancelled, explicit):
    scheduler = load('robotwin_scheduler', 'schedule_groundingpi_eval.py')
    run = tmp_path / 'run'
    (run / 'checkpoints').mkdir(parents=True)
    (run / 'checkpoints/steps_2000_pytorch_model.pt').write_bytes(b'checkpoint')
    (run / 'summary.jsonl').write_text('{"steps":2000}\n')
    for filename in ['config.yaml', 'dataset_statistics.json']:
        (run / filename).write_text('{}')
    template = tmp_path / 'deploy_224.yml'
    template.write_text('image_size: [224, 224]\ninclude_state: true\n')
    robotwin = tmp_path / 'RoboTwin'
    (robotwin / 'script').mkdir(parents=True)
    (robotwin / 'script/eval_policy.py').touch()
    args = SimpleNamespace(run_dir=run, gpus='0,1,2,3,4,5,6,7', episodes=20,
                           delay_hours=0, poll_seconds=.001, idle_memory_mib=4096,
                           starvla_python=Path('/bin/true'), robotwin_python=Path('/bin/true'),
                           robotwin_path=robotwin, output=tmp_path / 'output', dry_run=False,
                           deploy_template=template)
    if explicit:
        (run / 'final_model').mkdir()
        args.checkpoint = run / 'final_model/pytorch_model.pt'
        args.checkpoint.write_bytes(b'final checkpoint')
        (run / 'config.full.yaml').write_text('trainer:\n  max_train_steps: 10000\n')
    monkeypatch.setattr(scheduler, 'parse_args', lambda: args)
    monkeypatch.setattr(scheduler, 'gpu_memory', lambda gpus: {gpu: 0 for gpu in gpus})
    monkeypatch.setattr(scheduler, 'training_running', lambda _: False)
    monkeypatch.setattr(scheduler.signal, 'signal', lambda *_: None)
    calls = []
    handlers = {}
    monkeypatch.setattr(scheduler.signal, 'signal', lambda sig, handler: handlers.update({sig: handler}))

    class FakeProcess:
        def __init__(self, command, env, **kwargs):
            self.pid = 12345
            self.waiting = False
            self.wait_calls = 0
            calls.append((command, dict(env)))
            mode = command[command.index('-m') + 1]
            name = command[command.index('-n') + 1]
            if mode == 'demo_randomized':
                for task in scheduler.task_names():
                    result_dir = Path(env['ROBOTWIN_EVAL_RESULT_ROOT']) / task / 'model2robotwin_interface' / mode / name / 'timestamp'
                    result_dir.mkdir(parents=True)
                    (result_dir / '_result.txt').write_text('Timestamp: test\n\nInstruction Type: unseen\n\n0.5')

        def poll(self):
            return None

        def wait(self, timeout=None):
            assert not self.waiting, 'Reentrant Popen.wait during cancellation'
            self.waiting = True
            self.wait_calls += 1
            try:
                if cancelled and self.wait_calls == 1:
                    handlers[scheduler.signal.SIGTERM](scheduler.signal.SIGTERM, None)
                return 1 if len(calls) == 1 else 0
            finally:
                self.waiting = False

    monkeypatch.setattr(scheduler.subprocess, 'Popen', FakeProcess)
    if cancelled:
        monkeypatch.setattr(scheduler.os, 'killpg', lambda *_: None)
        with pytest.raises(SystemExit) as exc:
            scheduler.main()
        assert exc.value.code == 128 + scheduler.signal.SIGTERM
        assert len(calls) == 1
        assert json.loads((args.output / 'status.json').read_text())['status'] == 'cancelled'
        return
    assert scheduler.main() == 1
    assert [cmd[cmd.index('-m') + 1] for cmd, env in calls] == ['demo_clean', 'demo_randomized']
    assert calls[0][0][calls[0][0].index('-c') + 1] == calls[1][0][calls[1][0].index('-c') + 1]
    if explicit:
        assert calls[0][0][calls[0][0].index('-c') + 1] == str(args.checkpoint)
    assert all(env['ROBOTWIN_TEST_NUM'] == '20' and env['CUDA_VISIBLE_DEVICES'] == args.gpus for cmd, env in calls)
    assert all(env['DEPLOY_POLICY_TEMPLATE_PATH'] == str(args.output / 'deploy_policy.yml') for cmd, env in calls)
    assert all(env['ROBOTWIN_EVAL_RESULT_ROOT'] == str(args.output / 'results') for cmd, env in calls)
    assert (args.output / 'deploy_policy.yml').read_text() == template.read_text()
    manifest = json.loads((args.output / 'manifest.json').read_text())
    assert manifest['result_root'] == str(args.output / 'results')
    state = json.loads((args.output / 'status.json').read_text())
    assert state['status'] == 'complete_with_failures'
    assert state['stages']['clean2clean']['summary']['completed_tasks'] == 0
    assert state['stages']['clean2rand']['summary']['episodes'] == 1000
    assert state['stages']['clean2rand']['summary']['successes'] == 500
    report = (args.output / 'evaluation_summary.md').read_text()
    assert '| clean2rand | 50/50 | 500/1000 | 50.00% |' in report
    assert 'MISSING (infrastructure failure)' in report
    with pytest.raises(FileExistsError, match='already started'):
        scheduler.main()
    assert len(calls) == 2


def test_summary_legacy_result_root(tmp_path):
    scheduler = load('robotwin_scheduler_legacy', 'schedule_groundingpi_eval.py')
    result = tmp_path / 'eval_result/task/model2robotwin_interface/demo_clean/run/time/_result.txt'
    result.parent.mkdir(parents=True)
    result.write_text('0.75\n')
    stage = tmp_path / 'stage'
    stage.mkdir()
    summary = scheduler.summarize(stage, ['task'], 20, tmp_path, 'demo_clean', 'run')
    assert summary['successes'] == 15
    assert summary['episodes'] == 20


@pytest.mark.parametrize('image_size', [224, 448])
def test_groundingpi_observation_and_action_order(monkeypatch, image_size):
    adapter = load('robotwin_adapter', 'model2robotwin_interface.py')
    requests = []

    class FakeClient:
        def __init__(self, *args):
            pass

        def get_server_metadata(self):
            return {'action_chunk_size': 16}

        def predict_action(self, request):
            requests.append(request)
            return {'data': {'actions': np.tile(np.arange(14), (1, 16, 1))}}

    monkeypatch.setattr(adapter, 'WebsocketClientPolicy', FakeClient)
    client = adapter.get_model({'policy_ckpt_path': 'test.pt', 'image_size': [image_size, image_size],
                                'include_state': True, 'unnorm_key': None})
    state = np.arange(14, dtype=np.float32)
    images = [np.full((10, 12, 3), value, dtype=np.uint8) for value in [10, 20, 30]]
    action = client.step({'state': state, 'lang': 'pick', 'image': images}, step=0)
    example = requests[0]['examples'][0]
    np.testing.assert_array_equal(example['state'], state[[0,1,2,3,4,5,7,8,9,10,11,12,6,13]][None])
    assert [image.shape for image in example['image']] == [(image_size, image_size, 3)] * 3
    assert [int(image[0,0,0]) for image in example['image']] == [10,20,30]
    np.testing.assert_array_equal(action, np.arange(14)[[0,1,2,3,4,5,12,6,7,8,9,10,11,13]])
    # Other policies retain the previous no-state, 224-pixel default.
    legacy = adapter.get_model({'policy_ckpt_path': 'legacy.pt'})
    legacy.step({'state': state, 'lang': 'pick', 'image': images}, step=0)
    assert 'state' not in requests[-1]['examples'][0]
    assert requests[-1]['examples'][0]['image'][0].shape == (224,224,3)
