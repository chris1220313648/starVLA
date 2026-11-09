"""Regression checks for the v3 adapter and LIBERO observation contract."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'examples/simBenchmarks/LIBERO/train_files/groundingpi_libero.py'


def load_runner():
    spec = importlib.util.spec_from_file_location('groundingpi_libero', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_restores_state_contract_from_full_training_config(tmp_path):
    runner = load_runner()
    checkpoint = tmp_path / 'checkpoints' / 'steps_9000_pytorch_model.pt'
    cfg = {'framework': {'name': 'GroundingPIPI'}, 'datasets': {'vla_data': {'data_mix': 'libero_v3_four'}}}
    with pytest.raises(ValueError, match='include_state'):
        runner.evaluation_config(checkpoint, cfg)
    (tmp_path / 'config.full.yaml').write_text('datasets:\n  vla_data:\n    include_state: true\n    obs_image_size: [224, 224]\n')
    result = runner.evaluation_config(checkpoint, cfg)
    assert result['datasets']['vla_data']['include_state'] is True
    assert result['datasets']['vla_data']['obs_image_size'] == [224, 224]
    assert result['datasets']['vla_data']['data_mix'] == 'libero_v3_four'


def test_prepare_keeps_raw_metadata_and_local_cache(tmp_path):
    runner = load_runner()
    raw, adapter = tmp_path / 'raw', tmp_path / 'adapter'
    for suite in runner.SUITES[:4]:
        source = raw / suite
        (source / 'meta').mkdir(parents=True)
        (source / 'data').mkdir()
        (source / 'videos').mkdir()
        info = {'codebase_version': 'v3.0', 'total_episodes': 1, 'total_frames': 2,
                'features': {'observation.state': {'shape': [8]}, 'action': {'shape': [7]}}}
        (source / 'meta/info.json').write_text(json.dumps(info))
    args = argparse.Namespace(raw_data=raw, data_root=adapter, mix='four')
    runner.prepare(args)
    (adapter / runner.SUITES[0] / 'meta/stats_gr00t.json').write_text('cache')
    runner.prepare(args)
    for suite in runner.SUITES[:4]:
        assert not (raw / suite / 'meta/modality.json').exists()
        assert not (raw / suite / 'meta/stats_gr00t.json').exists()
        assert (adapter / suite / 'data').resolve() == raw / suite / 'data'
        assert not (adapter / suite / 'meta').is_symlink()
        modality = json.loads((adapter / suite / 'meta/modality.json').read_text())
        assert modality['state']['gripper']['end'] == 8
        assert modality['state']['x']['original_key'] == 'observation.state'
    args.data_root = raw
    with pytest.raises(ValueError, match='separate'):
        runner.prepare(args)


def test_v3_registry_and_action_normalization():
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
    data = ROBOT_TYPE_CONFIG_MAP['libero_v3_franka']
    assert data.action_indices == list(range(16))
    assert len(data.state_keys) == 8
    assert data.video_keys == ['video.primary_image', 'video.wrist_image']
    assert len(DATASET_NAMED_MIXTURES['libero_v3_four']) == 4
    assert len(DATASET_NAMED_MIXTURES['libero_v3_five']) == 5


def test_saved_normalization_roundtrip_and_raw_state(monkeypatch):
    from deployment.model_server import policy_norm_processor as norm
    stats = {}
    for key, dim in [('action', 7), ('state', 8)]:
        stats[key] = {'min': [-2.] * dim, 'max': [4.] * dim,
                      'mean': [1.] * dim, 'std': [1.] * dim,
                      'q01': [-1.9] * dim, 'q99': [3.9] * dim}
    cfg = {'datasets': {'vla_data': {'data_mix': 'libero_v3_four'}}}
    monkeypatch.setattr(norm, 'read_mode_config', lambda _: (cfg, {'franka': stats}))
    proc = norm.PolicyNormProcessor('unused')
    actions = np.tile(np.array([0., 1., 2., 3., -1., 0.5, 1.], dtype=np.float32), (16, 1))
    raw = {key: actions[:, i:i+1].copy() for i, key in enumerate(proc.action_keys)}
    state = np.arange(8, dtype=np.float32)[None]
    raw.update({key: state[:, i:i+1].copy() for i, key in enumerate(proc.state_keys)})
    transformed = proc.transform(raw)
    np.testing.assert_array_equal(np.concatenate([transformed[key] for key in proc.state_keys], -1), state)
    normalized = np.concatenate([np.asarray(transformed[key]) for key in proc.action_keys], axis=-1)
    np.testing.assert_allclose(proc.unapply_actions(normalized), actions, atol=1e-6)
    np.testing.assert_array_equal(normalized[:, -1], actions[:, -1])


def test_trainer_rejects_nonfinite_loss_before_optimizer_step(monkeypatch):
    import contextlib
    from types import SimpleNamespace
    import torch
    from torch import nn
    monkeypatch.setenv('STARVLA_DISABLE_DEEPSPEED', '1')
    from starVLA.training.train_starvla import VLATrainer

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, value):
            return {'action_loss': self.weight * value}

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.model = Policy()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(gradient_clipping=None))
    trainer.accelerator = SimpleNamespace(accumulate=lambda _: contextlib.nullcontext(),
                                         backward=lambda loss: loss.backward(), sync_gradients=True)
    with pytest.raises(FloatingPointError, match='non-finite loss'):
        trainer._train_step(torch.tensor(float('nan')))
    assert trainer.model.weight.item() == 1.0
    metrics = trainer._train_step(torch.tensor(1.0))
    assert trainer.model.weight.item() == pytest.approx(0.9)
    assert metrics['action_dit_loss'] == 1.0


def test_training_loop_enables_pretrained_backbone_training(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from torch import nn
    monkeypatch.setenv('STARVLA_DISABLE_DEEPSPEED', '1')
    from starVLA.training.train_starvla import VLATrainer

    trainer = VLATrainer.__new__(VLATrainer)
    # Reproduce a training framework containing an eval-mode HF backbone.
    backbone = nn.Sequential(nn.Linear(2, 2), nn.Dropout()).eval()
    trainer.model = nn.Sequential(backbone, nn.Linear(2, 1))
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(max_train_steps=0))
    trainer.accelerator = SimpleNamespace(is_local_main_process=False)
    trainer.completed_steps = 0
    trainer._log_training_config = Mock()
    trainer._create_data_iterators = Mock()
    trainer._finalize_training = Mock()
    assert not backbone.training
    trainer.train()
    assert all(module.training for module in trainer.model.modules())
