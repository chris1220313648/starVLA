from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from starVLA.model.modules.action_model.oat_ActionHeader import OATCodec, OATNormalization, oat_token_constraint
from starVLA.dataloader.u0_sequence import SequenceLiberoDataset
from test_u0_sequence import packer

ASSETS = Path(__file__).resolve().parents[1] / 'playground/Pretrained_models/OAT/libero/pretrained_model'


def test_oat_codec_normalization_and_frozen_precision():
    norm = OATNormalization(ASSETS)
    raw = np.zeros((2, 32, 7), dtype=np.float32)
    raw[1, :, 6] = 1
    x = norm.normalize(raw)
    assert np.all(x[0, :, 6] == 1) and np.all(x[1, :, 6] == -1)
    np.testing.assert_allclose(norm.unnormalize(x), raw, atol=1e-6)
    codec = OATCodec(ASSETS).train()
    assert not codec.training and not any(p.requires_grad for p in codec.parameters())
    a, b = codec.encode(x), codec.encode(x)
    assert a == b and np.asarray(a).shape == (2, 16)
    assert codec.decode(a).shape == (2, 32, 7)
    for bad in [np.zeros((2, 15), dtype=int), np.full((1, 16), 1920), np.zeros((1, 16), dtype=float)]:
        with pytest.raises(ValueError): codec.decode(bad)


def test_oat_sequence_mask_generation_and_boundaries():
    p = packer()
    p.interface.action_token_ids = list(range(149595, 151515))
    p.contract.update(action_horizon=32, action_token_length=16)
    e = dict(lang='pick cup', sequence_state=np.zeros((2,8)), sequence_image_codes=np.zeros((2,2,14,14),dtype=np.int32))
    tokens = [list(range(16)), list(range(16,32))]
    batch = p.training_inputs([e], [tokens])
    labels = batch['labels'][0]
    assert ((labels >= 149595) & (labels < 151515)).sum() == 32
    assert (labels >= 151854).sum() == 392
    history = [dict(state=e['sequence_state'][0],image_codes=e['sequence_image_codes'][0],action_tokens=tokens[0])]
    prompt = p.prompt(e['lang'], e['sequence_state'][1],e['sequence_image_codes'][1],history)
    assert batch['input_ids'][0,:len(prompt)].tolist() == prompt
    with pytest.raises(ValueError): p.action([0]*15)
    ds=SimpleNamespace(data_cfg={'sequence_h':2,'sequence_action_horizon':32},trajectory_ids=[0,1,2],trajectory_lengths=[63,64,66])
    assert SequenceLiberoDataset._get_all_steps(ds) == [(1,0),(2,0),(2,1),(2,2)]
    fn = oat_token_constraint(p.interface.action_token_ids, p.end)
    assert fn([]) == p.interface.action_token_ids
    assert fn([149595]*16) == [p.end]
    with pytest.raises(ValueError): fn([151515])
