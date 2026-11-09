import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from starVLA.model.framework.VLM4A import GroundingPIPI05 as pi05


@pytest.fixture
def model(monkeypatch):
    torch.manual_seed(42)
    class Interface(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = nn.Module()
            self.model.model = nn.Module()
            self.model.model.language_model = Qwen3Model(Qwen3Config(
                hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                vocab_size=16, attention_dropout=0.0))
    monkeypatch.setattr(pi05, 'GroundingPIInterface', Interface)
    cfg = OmegaConf.create({'framework': {'action_model': {
        'action_horizon': 3, 'action_dim': 2, 'state_dim': 3,
        'expert_width': 16, 'expert_mlp_dim': 32}}})
    return pi05.GroundingPIPI05(cfg)


def inputs():
    return (torch.randn(2, 5, 32), torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.bool),
            torch.randn(2, 3, 2), torch.tensor([.3, .7]))


def test_state_prompt():
    assert 'State: 0 128 255;' in pi05.state_prompt('pick_up\n cup', [-1, 0, 1], 3)
    for state in ([2, 0, 1], [np.nan, 0, 1], [0, 1]):
        with pytest.raises(ValueError):
            pi05.state_prompt('pick', state, 3)


def test_cache_matches_joint_and_is_immutable(model):
    model.eval()
    prefix, mask, noise, time = inputs()
    with torch.no_grad():
        full = model.velocity(prefix, mask, noise, time)
        cache = model.prefix_cache(prefix, mask)
        snapshots = [(k.clone(), v.clone()) for k, v in cache]
        cached = model.velocity(None, mask, noise, time, cache)
        torch.testing.assert_close(full, cached)
        model.velocity(None, mask, noise, time / 2, cache)
        for (k, v), (old_k, old_v) in zip(cache, snapshots):
            torch.testing.assert_close(k, old_k, rtol=0, atol=0)
            torch.testing.assert_close(v, old_v, rtol=0, atol=0)
            assert k.shape[-2] == prefix.shape[1]


def test_padding_and_time_condition(model):
    model.eval()
    prefix, mask, noise, time = inputs()
    baseline = model.velocity(prefix, mask, noise, time)
    changed = prefix.clone()
    changed[~mask] = 1000
    torch.testing.assert_close(baseline, model.velocity(changed, mask, noise, time))
    assert not torch.allclose(baseline, model.velocity(prefix, mask, noise, time + .1))
    changed = prefix.clone()
    changed[mask] += torch.randn_like(changed[mask]) * 5
    assert not torch.allclose(baseline, model.velocity(changed, mask, noise, time))


def test_prefix_cannot_read_actions_and_is_bidirectional(model):
    model.eval()
    prefix, mask, noise, time = inputs()
    suffix, _, _, cond = model.action_model.embed_action_suffix(None, noise, time)
    positions, action_pos = model._positions(mask)
    p1, _ = model._joint_layer(0, prefix, suffix, cond, mask, positions, action_pos)
    p2, _ = model._joint_layer(0, prefix, suffix * 100, cond, mask, positions, action_pos)
    torch.testing.assert_close(p1, p2, atol=0, rtol=0)
    altered = prefix.clone()
    altered[:, -1] += torch.randn_like(altered[:, -1]) * 10
    p3, _, _ = model._prefix_layer(0, altered, mask, positions)
    assert not torch.allclose(p1[:, 2], p3[:, 2])


def test_checkpointed_gradients_and_parameter_updates(model):
    model.train()
    prefix, mask, noise, time = inputs()
    before = {k: p.detach().clone() for k, p in model._watched.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Tiny synthetic prefix bypasses embedding/vision but exercises the real Qwen layers.
    for _ in range(3):
        opt.zero_grad()
        loss = (model.velocity(prefix, mask, noise, time) - 1).square().mean()
        loss.backward()
        opt.step()
    for name, parameter in model._watched.items():
        assert model._gradient_norms[name] > 0, name
        assert not torch.equal(before[name], parameter), name
    state = model.state_dict()
    model.load_state_dict(state, strict=True)
    assert not any('lm_head.weight' in key for key in state)


def test_pi05_data_contract_and_lr_groups(model):
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
    mix = DATASET_NAMED_MIXTURES['groundingpi_pi05_four']
    assert len(mix) == 4 and all(row[2] == 'groundingpi_pi05_franka' for row in mix)
    config = ROBOT_TYPE_CONFIG_MAP['groundingpi_pi05_franka']
    assert config.action_indices == list(range(16))
    cfg = OmegaConf.load('examples/simBenchmarks/LIBERO/train_files/groundingpipi05.yaml')
    groups = {g['name']: g for g in build_param_lr_groups(model, cfg)}
    assert groups['qwen_vl_interface']['lr'] == 1e-5
    assert groups['action_model']['lr'] == 1e-4
    params = [id(p) for group in groups.values() for p in group['params']]
    assert len(params) == len(set(params))
    assert set(params) == {id(p) for p in model.parameters() if p.requires_grad}


def test_prefix_matches_native_qwen_layer(model):
    model.eval()
    prefix, mask, _, _ = inputs()
    positions, _ = model._positions(mask)
    layer = model.language_model.layers[0]
    layer.self_attn.config._attn_implementation = 'sdpa'
    additive = torch.where(mask[:, None, None, :], 0.0, float('-inf'))
    native = layer(prefix, attention_mask=additive,
                   position_embeddings=model.language_model.rotary_emb(prefix, positions))
    actual, _, _ = model._prefix_layer(0, prefix, mask, positions)
    torch.testing.assert_close(actual, native)


def test_pi05_training_serving_state_normalization(monkeypatch):
    from deployment.model_server import policy_norm_processor as norm
    stats = {kind: {name: value for name, value in {
        'min': [-2.] * dim, 'max': [4.] * dim, 'mean': [1.] * dim,
        'std': [1.] * dim, 'q01': [-1.9] * dim, 'q99': [3.9] * dim}.items()}
        for kind, dim in [('action', 7), ('state', 8)]}
    cfg = {'datasets': {'vla_data': {'data_mix': 'groundingpi_pi05_four'}}}
    monkeypatch.setattr(norm, 'read_mode_config', lambda _: (cfg, {'franka': stats}))
    proc = norm.PolicyNormProcessor('unused')
    state = np.array([[-20, -2, 1, 4, 20, 0, 2, 3]], np.float32)
    expected = np.clip(2 * (state + 2) / 6 - 1, -1, 1)
    np.testing.assert_allclose(proc.apply_state(state), expected)
    raw = {k: state[:, i:i+1].copy() for i, k in enumerate(proc.state_keys)}
    train = proc.transform(raw)
    np.testing.assert_allclose(np.concatenate([train[k] for k in proc.state_keys], -1), expected)


def test_checkpoint_keeps_state_serving_contract(model):
    from starVLA.training.trainer_utils.config_tracker import wrap_config
    cfg = OmegaConf.create(OmegaConf.to_container(model.config))
    cfg.datasets.vla_data.include_state = True
    cfg.datasets.vla_data.normalize_state_on_server = True
    tracked = wrap_config(cfg)
    pi05.GroundingPIPI05(tracked)
    saved = tracked.export_accessed_config()
    assert saved['datasets']['vla_data']['include_state'] is True
    assert saved['datasets']['vla_data']['normalize_state_on_server'] is True
    cfg.datasets.vla_data.normalize_state_on_server = False
    with pytest.raises(ValueError, match='normalization'):
        pi05.GroundingPIPI05(cfg)


def test_serving_numpy_images_preserve_pixels_and_camera_order(model, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    arrays = [np.full((16, 24, 3), value, dtype=np.uint8) for value in (17, 219)]
    captured = []
    def pack(images, prompts):
        captured.append(images)
        assert all(isinstance(image, Image.Image) for image in images[0])
        for image, array in zip(images[0], arrays):
            assert image.size == (24, 16)
            np.testing.assert_array_equal(np.asarray(image), array)
        return SimpleNamespace(input_ids=torch.tensor([[1, 0]]),
                               attention_mask=torch.tensor([[1, 1]]),
                               pixel_values=torch.zeros(1), image_grid_thw=torch.ones(1, 3))
    core = model.qwen_vl_interface.model.model
    monkeypatch.setattr(model.qwen_vl_interface, 'build_qwenvl_inputs', pack, raising=False)
    monkeypatch.setattr(core, 'get_input_embeddings', lambda: nn.Embedding(2, 32), raising=False)
    monkeypatch.setattr(core, 'get_image_features', lambda *args: [torch.ones(1, 32)], raising=False)
    monkeypatch.setattr(core, 'get_placeholder_mask',
                        lambda ids, embs, **kw: ((ids == 1).unsqueeze(-1).expand_as(embs), None), raising=False)
    for images in (arrays, [Image.fromarray(array) for array in arrays]):
        prefix, mask = model.prepare_prefix([{'lang': 'pick', 'state': np.zeros(3), 'image': images}])
        assert prefix.shape == (1, 2, 32)
        torch.testing.assert_close(prefix[:, 0], torch.ones(1, 32))
    assert len(captured) == 2
    for array, value in zip(arrays, (17, 219)):
        assert np.all(array == value)


@pytest.fixture
def ki_model(model, monkeypatch):
    from types import SimpleNamespace
    from transformers.feature_extraction_utils import BatchFeature
    model.knowledge_insulation = True
    model.language_model.requires_grad_(True)
    model.qwen_vl_interface.model.lm_head = nn.Linear(32, 16, bias=False)
    embedding = nn.Embedding(16, 32)
    model.qwen_vl_interface.model.model.embedding = embedding
    model.fast_tokenizer = SimpleNamespace(encoder_action2fastoken=lambda actions: [[3, 4] for _ in actions])
    def pack(images, prompts, solutions=None):
        assert solutions is not None
        b = len(prompts)
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]]).expand(b, -1)
        mask = torch.tensor([[0, 1, 1, 1, 1, 1]]).expand(b, -1)
        labels = torch.tensor([[-100, -100, -100, 3, 4, 5]]).expand(b, -1)
        return BatchFeature(data=dict(input_ids=ids, attention_mask=mask, labels=labels))
    monkeypatch.setattr(model.qwen_vl_interface, 'build_qwenvl_inputs', pack, raising=False)
    monkeypatch.setattr(model, '_embed_inputs', lambda inputs: embedding(inputs.input_ids))
    return model


def ki_examples():
    from PIL import Image
    return [dict(lang='pick', state=np.zeros(3), image=[Image.new('RGB', (8, 8))],
                 action=np.zeros((3, 2), dtype=np.float32)) for _ in range(2)]


def test_ki_gradient_isolation(ki_model):
    model = ki_model.train()
    out = model(ki_examples())
    torch.testing.assert_close(out['total_loss'], out['action_loss'] + out['fast_token_loss'])
    out['action_loss'].backward()
    assert all(p.grad is None for p in model.qwen_vl_interface.parameters())
    assert model.action_model.action_out_proj.weight.grad.norm() > 0
    model.zero_grad(set_to_none=True)
    out = model(ki_examples())
    out['fast_token_loss'].backward()
    assert all(p.grad is None for p in model.action_model.parameters())
    assert model.language_model.layers[0].self_attn.k_proj.weight.grad.norm() > 0
    assert model.language_model.layers[-1].mlp.up_proj.weight.grad.norm() > 0
    assert model.qwen_vl_interface.model.lm_head.weight.grad.norm() > 0


def test_ki_answers_cannot_leak_to_fm(ki_model, monkeypatch):
    model = ki_model.eval()
    noise, time = torch.randn(2, 3, 2), torch.tensor([.3, .7])
    original = model.qwen_vl_interface.build_qwenvl_inputs
    with torch.no_grad():
        before = model(ki_examples(), noise=noise, time=time)
        def changed(*a, **kw):
            inputs = original(*a, **kw)
            inputs.input_ids = inputs.input_ids.clone()
            inputs.input_ids[:, 3:] = torch.tensor([8, 9, 10])
            inputs.labels = inputs.labels.clone()
            inputs.labels[:, 3:] = torch.tensor([8, 9, 10])
            return inputs
        monkeypatch.setattr(model.qwen_vl_interface, 'build_qwenvl_inputs', changed)
        after = model(ki_examples(), noise=noise, time=time)
    torch.testing.assert_close(before['action_loss'], after['action_loss'], rtol=0, atol=0)
    assert not torch.allclose(before['fast_token_loss'], after['fast_token_loss'])


def test_ki_attention_and_ce_padding(ki_model):
    mask = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.bool)
    labels = torch.tensor([[-100, -100, -100, 3, 4]])
    attention, prefix = ki_model.ki_attention(mask, labels)
    assert prefix.tolist() == [[False, True, True, False, False]]
    assert attention[0, 0, 1].tolist() == [False, True, True, False, False]
    assert attention[0, 0, 3].tolist() == [False, True, True, True, False]
    hidden = torch.randn(1, 5, 32)
    loss = ki_model._fast_ce(hidden, labels)
    # The two preceding hidden positions predict the two target tokens.
    reference = torch.nn.functional.cross_entropy(
        ki_model.qwen_vl_interface.model.lm_head(ki_model.language_model.norm(hidden[:, 2:4])).flatten(0, 1), torch.tensor([3, 4]))
    torch.testing.assert_close(loss, reference)


def test_ki_combined_loss_updates_both_groups(ki_model):
    model = ki_model.train()
    before = {name: p.detach().clone() for name, p in [
        ('backbone', model.language_model.layers[0].self_attn.k_proj.weight),
        ('expert', model.action_model.action_out_proj.weight),
        ('head', model.qwen_vl_interface.model.lm_head.weight)]}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(ki_examples())['total_loss'].backward()
    opt.step()
    for name, p in [('backbone', model.language_model.layers[0].self_attn.k_proj.weight),
                    ('expert', model.action_model.action_out_proj.weight),
                    ('head', model.qwen_vl_interface.model.lm_head.weight)]:
        assert not torch.equal(p, before[name])


def test_continuous_expert_state_cache_and_gradients(model):
    cfg = OmegaConf.create({'framework': {'action_model': {
        'action_horizon': 3, 'action_dim': 2, 'state_dim': 3,
        'state_representation': 'continuous_expert_token',
        'expert_width': 16, 'expert_mlp_dim': 32}}})
    continuous = pi05.GroundingPIPI05(cfg)
    prefix, mask, noise, time = inputs()
    state = torch.randn(2,3)
    continuous.eval()
    full = continuous.velocity(prefix,mask,noise,time,state=state)
    cache = continuous.prefix_cache(prefix,mask)
    cached = continuous.velocity(None,mask,noise,time,cache,state=state)
    torch.testing.assert_close(full,cached)
    assert full.shape == noise.shape
    assert continuous._positions(mask)[1].shape == (2,4)
    changed = continuous.velocity(None,mask,noise,time,cache,state=state+1)
    assert not torch.allclose(full,changed)
    with pytest.raises(ValueError):continuous.velocity(None,mask,noise,time,cache)
    continuous.train()
    continuous.gradient_checkpointing = True
    loss = continuous.velocity(prefix,mask,noise,time,state=state).square().mean()
    loss.backward()
    assert continuous.action_model.state_encoder[0].weight.grad.abs().max() > 0
    continuous.load_state_dict(continuous.state_dict(),strict=True)


def test_continuous_state_mask_and_prompt(model, monkeypatch):
    cfg = OmegaConf.create({'framework': {'action_model': {
        'action_horizon': 3, 'action_dim': 2, 'state_dim': 3,
        'state_representation': 'continuous_expert_token',
        'expert_width': 16, 'expert_mlp_dim': 32}}})
    continuous = pi05.GroundingPIPI05(cfg).eval()
    prefix, mask, noise, time = inputs()
    monkeypatch.setattr(continuous, 'prepare_prefix', lambda examples:(prefix,mask))
    # time=1 makes the noisy input independent of targets: masked labels must not affect loss.
    examples = [{'state':np.zeros((1,3)), 'action':np.zeros((3,2)), 'action_mask':np.array([1,1,0],dtype=bool)} for _ in range(2)]
    baseline = continuous(examples, noise=noise, time=torch.ones(2))['action_loss']
    for e in examples:e['action'][2] = 100
    torch.testing.assert_close(baseline,continuous(examples,noise=noise,time=torch.ones(2))['action_loss'])
    cfg.framework.knowledge_insulation = True
    with pytest.raises(ValueError,match='knowledge insulation'):pi05.GroundingPIPI05(cfg)


def test_continuous_state_serving_contract_is_saved(model):
    from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig
    cfg=AccessTrackedConfig(OmegaConf.create({'framework':{'action_model':{
        'action_horizon':3,'action_dim':2,'state_dim':3,
        'state_representation':'continuous_expert_token','expert_width':16,'expert_mlp_dim':32}},
        'datasets':{'vla_data':{'include_state':True,'normalize_state_on_server':True,
        'obs_image_size':[384,320],'image_layout':'head_top_wrists_bottom'}}}))
    pi05.GroundingPIPI05(cfg)
    saved=cfg.export_accessed_config()
    assert saved['datasets']['vla_data']['obs_image_size']==[384,320]
    assert saved['framework']['action_model']['state_representation']=='continuous_expert_token'
