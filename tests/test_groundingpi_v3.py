import numpy as np
import torch
from torch import nn
from types import SimpleNamespace
from omegaconf import OmegaConf


def test_goal_state_and_action_contract(monkeypatch):
    from deployment.model_server import policy_norm_processor as norm
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
    assert DATASET_NAMED_MIXTURES['groundingpi_v3_libero_goal'] == [('libero_goal', 1.0, 'groundingpi_v3_goal_franka')]
    stats = {kind: {name: values for name, values in {
        'min': [-2.] * dim, 'max': [4.] * dim, 'mean': [1.] * dim,
        'std': [1.] * dim, 'q01': [-1.9] * dim, 'q99': [3.9] * dim}.items()}
        for kind, dim in [('action', 7), ('state', 8)]}
    cfg = {'datasets': {'vla_data': {'data_mix': 'groundingpi_v3_libero_goal'}}}
    monkeypatch.setattr(norm, 'read_mode_config', lambda _: (cfg, {'franka': stats}))
    proc = norm.PolicyNormProcessor('unused')
    state = np.array([[-20, -2, 1, 4, 20, 0, 2, 3]], np.float32)
    expected = np.clip(2 * (state + 2) / 6 - 1, -1, 1)
    np.testing.assert_allclose(proc.apply_state(state), expected)
    raw = {k: state[:, i:i+1].copy() for i,k in enumerate(proc.state_keys)}
    train = proc.transform(raw)
    np.testing.assert_allclose(np.concatenate([train[k] for k in proc.state_keys], -1), expected)
    actions = np.linspace(-1,1,112,dtype=np.float32).reshape(16,7)
    raw = {k: actions[:,i:i+1].copy() for i,k in enumerate(proc.action_keys)}
    transformed = proc.transform(raw)
    normalized = np.concatenate([transformed[k] for k in proc.action_keys], -1)
    np.testing.assert_allclose(proc.unapply_actions(normalized), actions, atol=1e-6)


def test_v3_architecture_and_learning_rate_groups(monkeypatch):
    import starVLA.model.framework.VLM4A.QwenPI_v3 as v3
    from starVLA.model.framework.VLM4A.GroundingPIPIV3 import GroundingPIPIV3
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 16)
            self.config = SimpleNamespace(hidden_size=16, text_config=SimpleNamespace(num_hidden_layers=2))
        def get_input_embeddings(self):
            return self.emb
    class Interface(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
    monkeypatch.setattr(v3, 'get_vlm_model', lambda config: Interface())
    cfg = OmegaConf.load('examples/simBenchmarks/LIBERO/train_files/groundingpipi_v3.yaml')
    cfg.framework.action_model.diffusion_model_cfg.action_dit_hidden_dim = 8
    cfg.framework.action_model.diffusion_model_cfg.attention_head_dim = 4
    model = GroundingPIPIV3(cfg)
    assert isinstance(model.project_layers[0][0], nn.LayerNorm)
    assert model.project_layers[0][1].weight.shape == (8,16)
    assert model.action_model.state_encoder is None
    assert not model.action_model.model._interleave_self_attention
    assert model.state2str_transform(np.array([-1,0,1])) == '0 128 255'
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
    groups = build_param_lr_groups(model, cfg)
    by_name = {g['name']: g for g in groups}
    assert by_name['project_layers']['lr'] == 1e-4
    assert by_name['qwen_vl_interface']['lr'] == 1e-5
    assert by_name['action_model']['lr'] == 1e-4
    params = [id(p) for g in groups for p in g['params']]
    assert len(params) == len(set(params)) == len(list(model.parameters()))


def test_serving_normalizes_state_once_without_mutating_request():
    from unittest.mock import Mock
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
    wrapper._default_unnorm_key = 'franka'
    wrapper._model_cfg = {'datasets': {'vla_data': {'normalize_state_on_server': True}}}
    proc = Mock()
    proc.apply_state.side_effect = lambda state: state / 10
    proc.unapply_actions.side_effect = lambda actions: actions
    wrapper._get_processor = Mock(return_value=proc)
    wrapper._framework = Mock()
    wrapper._framework.predict_action.return_value = {'normalized_actions': np.zeros((1,16,7))}
    example = {'state': np.ones((1,8), dtype=np.float32)*5}
    wrapper.predict_action([example])
    assert proc.apply_state.call_count == 1
    supplied = wrapper._framework.predict_action.call_args.kwargs['examples'][0]
    np.testing.assert_array_equal(supplied['state'], np.full((1,8),.5))
    np.testing.assert_array_equal(example['state'], np.full((1,8),5))
    wrapper._model_cfg = {}
    wrapper.predict_action([example])
    assert proc.apply_state.call_count == 1
