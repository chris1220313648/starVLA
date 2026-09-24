from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from starVLA.model.modules.vlm.U0 import U0Interface, action_cross_entropy
from starVLA.model.modules.vlm.u0_sequence import SequencePacker, state_bins
from starVLA.model.modules.vlm.u0.tokenization_unis import UNISTokenizer
from starVLA.dataloader.u0_sequence import SequenceLiberoDataset


def packer():
    interface = U0Interface.__new__(U0Interface)
    torch.nn.Module.__init__(interface)
    base = '/root/nas/code/Xiaomi-Robotics-U0/training/models/Xiaomi-Robotics-U0-4B'
    interface.tokenizer = UNISTokenizer(vocab_file=base+'/unis.tiktoken', special_tokens_file=base+'/unis_vision_tokens.txt')
    interface._set_special_tokens()
    interface.image_size, interface.max_length = 224, 2048
    interface.action_token_ids = list(range(149595,151643))
    interface.model = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(1)))
    contract = dict(sequence_h=2, action_horizon=8, state_dim=8,
        state_statistics=dict(q01=[0]*8, q99=[1]*7+[0]),
        prompt_template='Instruction: {instruction} State: {state} ')
    return SequencePacker(interface, contract)


def test_sequence_mask_history_and_boundaries():
    p = packer()
    codes = np.zeros((2,2,14,14),dtype=np.int32)
    e = dict(lang='pick cup', sequence_state=np.zeros((2,8)), sequence_image_codes=codes)
    tokens = [[1,2], [3,4,5]]
    batch = p.training_inputs([e], [tokens])
    ids, labels = batch['input_ids'][0].tolist(), batch['labels'][0]
    assert int((labels >= 151854).sum()) == 392
    assert int(labels.ne(-100).sum()) == 392+5+2+1
    history = [dict(state=e['sequence_state'][0],image_codes=codes[0], fast_tokens=tokens[0])]
    prompt = p.prompt(e['lang'], e['sequence_state'][1], codes[1], history)
    assert ids[:len(prompt)] == prompt
    assert ids[len(prompt):] == p.action(tokens[1]) + [p.tok.eos_token_id]
    np.testing.assert_array_equal(state_bins([0, .5, 1, -5, 5, .5, .5, 99], p.contract['state_statistics']), [0,128,255,0,255,128,128,128])
    with pytest.raises(ValueError): p.prompt('x', np.zeros(8), codes[0], history*2)
    with pytest.raises(ValueError): p.text('<|extra_100|>')
    reserved = p.tok.decode([149595])
    with pytest.raises(ValueError): p.text(reserved)
    p.interface.max_length=10
    with pytest.raises(ValueError,match='truncation'): p.training_inputs([e],[tokens])
    ds=SimpleNamespace(data_cfg={'sequence_h':2},trajectory_ids=[0,1,2],trajectory_lengths=[15,16,18])
    assert SequenceLiberoDataset._get_all_steps(ds)==[(1,0),(2,0),(2,1),(2,2)]
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
    valid=SequenceLiberoDataset._get_all_steps(ds)
    windows=SimpleNamespace(all_steps=valid)
    windows.sample_window=lambda rng: SequenceLiberoDataset.sample_window(windows,rng)
    mixture=SimpleNamespace(mode='train',epoch=0,seed=42,datasets=[windows],dataset_sampling_weights=[1.])
    for i in range(100):
        _,ep,t=LeRobotMixtureDataset.sample_step(mixture,i)
        assert (ep,t) in valid


def test_grouped_loss_and_gradients():
    torch.manual_seed(0)
    h=torch.randn(1,5,2,requires_grad=True)
    head=torch.nn.Linear(2,282926,bias=False)
    labels=torch.tensor([[-100,149595,151747,151854,151850]])
    out=action_cross_entropy(h,labels,head,2,1.0)
    ce=torch.nn.functional.cross_entropy(head(h[:,:-1]).reshape(-1,282926),labels[:,1:].flatten(),reduction='none')
    expected=ce[[0,1,3]].mean()+ce[2]
    torch.testing.assert_close(out['action_loss'],expected)
    actual=torch.autograd.grad(out['action_loss'],(h,head.weight),retain_graph=True)
    wanted=torch.autograd.grad(expected,(h,head.weight))
    for a,b in zip(actual,wanted): torch.testing.assert_close(a,b)


def test_client_history_and_reset():
    from examples.simBenchmarks.LIBERO.eval_files import model2libero_interface as m
    class Fake:
        def __init__(self,*a): self.calls=[]
        def get_server_metadata(self): return dict(action_chunk_size=8,sequence_h=2,requires_raw_state=True)
        def predict_action(self,x):
            self.calls.append(x)
            return {'data':{'actions':np.zeros((1,8,7)), 'fast_tokens':[[1,2,3]]}}
    with patch.object(m,'WebsocketClientPolicy',Fake):
        c=m.ModelClient(image_size=None,action_ensemble=False)
        e=dict(image=[np.zeros((256,256,3),dtype=np.uint8)]*2,state=np.arange(8),lang='task')
        for t in range(17): c.step(e,t)
        assert [len(x['examples'][0]['history']) for x in c.client.calls]==[0,1,1]
        assert c.client.calls[1]['examples'][0]['history'][0]['fast_tokens']==[1,2,3]
        c.reset('task');c.step(e,0)
        assert c.client.calls[-1]['examples'][0]['history']==[]
        c.step(e,16) # skipped steps must not commit a partial chunk
        assert c.client.calls[-1]['examples'][0]['history']==[]
