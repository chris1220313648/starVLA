"""Offline contracts for GroundingPI packing and action-framework integration."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from starVLA.model.modules.vlm.GroundingPI import GroundingPIInterface, expand_image_tokens


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, texts, **kwargs):
        self.texts = texts
        ids = [self.encode(text) for text in texts]
        length = max(map(len, ids))
        return dict(input_ids=torch.tensor([[0]*(length-len(row))+row for row in ids]),
                    attention_mask=torch.tensor([[0]*(length-len(row))+[1]*len(row) for row in ids]))


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.image_processor = Mock(merge_size=2)
        self.image_processor.side_effect = lambda images, **kw: {
            'image_grid_thw': torch.tensor([[1, 2, 2*(i+1)] for i in range(len(images))]),
            'pixel_values': torch.zeros(len(images), 3),
        }

    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        text = ''
        for message in messages:
            text += message['role'] + ':'
            for content in message['content']:
                text += '<|image_pad|>' if content['type'] == 'image' else content['text']
            text += '|end|'
        return text + ('assistant:' if add_generation_prompt else '')


@pytest.fixture
def interface():
    obj = GroundingPIInterface.__new__(GroundingPIInterface)
    nn.Module.__init__(obj)
    obj.model = Mock(device=torch.device('cpu'))
    obj.processor = Processor()
    obj.config = OmegaConf.create({'datasets': {'vla_data': {}}})
    return obj


def test_expand_image_tokens():
    assert expand_image_tokens('a<|image_pad|>b<|image_pad|>c', [2, 1]).count('<|image_pad|>') == 3
    for counts in ([1], [0, 1], [1, 1, 1]):
        with pytest.raises(ValueError):
            expand_image_tokens('a<|image_pad|>b<|image_pad|>c', counts)


def test_multiview_batch_order_and_padding(interface):
    output = interface.build_qwenvl_inputs([[1], [2, 3]], ['short', 'long instruction'])
    assert [text.count('<|image_pad|>') for text in interface.processor.tokenizer.texts] == [1, 5]
    assert interface.processor.image_processor.call_args.kwargs['images'] == [1, 2, 3]
    assert output['attention_mask'][0, 0] == 0
    assert 'labels' not in output


def test_labels_cover_only_assistant_and_eos(interface):
    output = interface.build_qwenvl_inputs([[1], [2, 3]], ['short', 'long instruction'], ['ACTION', 'OTHER'])
    for row, expected in zip(output['labels'], ['ACTION|end|', 'OTHER|end|']):
        assert ''.join(chr(i) for i in row.tolist() if i != -100) == expected
    assert (output['labels'][output['attention_mask'] == 0] == -100).all()


def test_bad_batch_and_grid_fail(interface):
    with pytest.raises(ValueError):
        interface.build_qwenvl_inputs([[1]], [])
    interface.processor.image_processor.side_effect = None
    interface.processor.image_processor.return_value = {'image_grid_thw': torch.tensor([[1, 1, 3]])}
    with pytest.raises(ValueError, match='divisible'):
        interface.build_qwenvl_inputs([[1]], ['test'])


def test_supervised_forward_keeps_logits(interface):
    interface.forward(labels=torch.tensor([[1, 2]]))
    assert 'logits_to_keep' not in interface.model.call_args.kwargs
    interface.forward(output_hidden_states=True)
    assert interface.model.call_args.kwargs['logits_to_keep'] == 1


def test_registration_and_defaults():
    from starVLA.model.framework.VLM4A._groundingpi_config import grounding_config
    from starVLA.model.tools import FRAMEWORK_REGISTRY
    from starVLA.model.framework.base_framework import _auto_import_framework_modules
    _auto_import_framework_modules()
    for name in ['GroundingPIPI', 'GroundingPIOFT', 'GroundingPIFast']:
        assert name in FRAMEWORK_REGISTRY.list()
        assert FRAMEWORK_REGISTRY[name].__module__ == f"starVLA.model.framework.VLM4A.{name}"
        cfg = grounding_config(None, name)
        assert cfg.framework.qwenvl.backbone == 'groundingpi'
        assert cfg.framework.action_model.action_horizon == 16
    cfg = grounding_config({'framework': {'action_model': {'future_action_window_size': 31}}}, 'GroundingPIPI')
    assert cfg.framework.action_model.action_horizon == 32


def test_explicit_backbone_dispatch(monkeypatch):
    from starVLA.model.modules.vlm import get_vlm_model
    factory = Mock(return_value='grounding')
    monkeypatch.setattr('starVLA.model.modules.vlm.GroundingPI.GroundingPIInterface', factory)
    cfg = OmegaConf.create({'framework': {'qwenvl': {'base_vlm': '/weights/renamed', 'backbone': 'groundingpi'}}})
    assert get_vlm_model(cfg) == 'grounding'


def test_tracked_config_keeps_framework_updates():
    from starVLA.model.framework.VLM4A._groundingpi_config import grounding_config
    from starVLA.training.trainer_utils.config_tracker import wrap_config
    original = wrap_config(OmegaConf.create({'framework': {'name': 'GroundingPIPI'}}))
    cfg = grounding_config(original, 'GroundingPIPI')
    assert cfg is original
    cfg.framework.qwenvl.vl_hidden_dim = 2560
    assert original.unwrap().framework.qwenvl.vl_hidden_dim == 2560


def test_fast_ignores_prompt_actions_and_rejects_empty_generation():
    import numpy as np
    from starVLA.model.framework.VLM4A.GroundingPIFast import GroundingPIFast
    model = GroundingPIFast.__new__(GroundingPIFast)
    nn.Module.__init__(model)
    model.config = OmegaConf.create({'framework': {'action_model': {'action_dim': 7}}})
    model.action_horizon = 16
    iface = Mock(_ACTION_TOKEN_MIN=100, _ACTION_TOKEN_MAX=102)
    iface.build_qwenvl_inputs.return_value = {'input_ids': torch.tensor([[100, 10]])}
    iface.generate.return_value = torch.tensor([[100, 10, 101, 102, 0]])
    model.qwen_vl_interface = iface
    decoder = Mock(return_value=np.zeros((1, 16, 7)))
    model.action_model = SimpleNamespace(fast_tokenizer=SimpleNamespace(
        decode=decoder, bpe_tokenizer=SimpleNamespace(decode=lambda ids: 'x' * 112)))
    sample = {'image': [], 'lang': 'test'}
    model.predict_action([sample], do_sample=False, use_ddim=True, num_ddim_steps=10)
    assert iface.generate.call_args.kwargs['do_sample'] is False
    assert 'use_ddim' not in iface.generate.call_args.kwargs
    assert 'num_ddim_steps' not in iface.generate.call_args.kwargs
    assert decoder.call_args.args[0] == [[1, 2]]
    iface.generate.return_value = torch.tensor([[100, 10, 0]])
    with pytest.raises(ValueError, match='no action tokens'):
        model.predict_action([sample])


def test_mixed_missing_assistant_answers(interface):
    output=interface.build_qwenvl_inputs([[1],[2]],['short','long instruction'],['ANSWER',None],allow_missing_solutions=True)
    assert output.labels[1].eq(-100).all()
    assert output.prefix_mask[1].equal(output.attention_mask[1].bool())
    assert output.prefix_mask.equal(output.attention_mask.bool() & output.labels.eq(-100))
    expected='ANSWER|end|'
    assert ''.join(chr(i) for i in output.labels[0].tolist() if i!=-100)==expected
    with pytest.raises(ValueError,match='Missing assistant'):
        interface.build_qwenvl_inputs([[1]],['short'],[None])
    empty=interface.build_qwenvl_inputs([[1]],['short'],[None],allow_missing_solutions=True)
    assert empty.labels.eq(-100).all() and empty.prefix_mask.equal(empty.attention_mask.bool())
