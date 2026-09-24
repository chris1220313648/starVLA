"""Explicit real-data checkpoint/cache verification; does not train."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.dataloader.u0_vision_cache import dataset_specs
from deployment.model_server.policy_wrapper import PolicyServerWrapper


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args=parser.parse_args()
    checkpoint=Path(args.checkpoint).resolve()
    root=checkpoint.parent.parent
    cfg=OmegaConf.load(root/'config.full.yaml')
    wrapper=PolicyServerWrapper(str(checkpoint), config_overrides=['framework.u0.gradient_checkpointing=false','framework.u0.use_cached_vision=false'])
    model=wrapper._framework
    interface=model.u0_interface
    name,_,robot=next(x for x in dataset_specs(cfg) if 'goal' in x[0])
    ds=make_LeRobotSingleDataset(Path(cfg.datasets.vla_data.data_root_dir),name,robot,data_cfg=cfg.datasets.vla_data)
    sample=ds[0]
    is_oat = model.sequence_packer.contract.get('codec') == 'oat'
    tokens = model.oat_codec.encode(sample['sequence_action']) if is_oat else model.action_model.encoder_action2fastoken(list(sample['sequence_action']))
    inputs=model.sequence_packer.training_inputs([sample],[tokens])
    def logits():
        with torch.inference_mode():
            hidden=interface.model.get_decoder()(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],use_cache=False).last_hidden_state
            return interface.model.get_output_embeddings()(hidden[:,-2]).float()
    before=logits()
    model.load_checkpoint_state_dict(load_file(str(checkpoint)))
    after=logits()
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    images=[]
    for segment in range(2):
        raw=ds.get_step_data(sample['trajectory_id'],sample['frame_index']+model.action_horizon*segment)
        views=[raw[key][0] for key in ds.modality_keys['video']]
        images.append(views)
        for j,image in enumerate(views):
            codes=interface.encode_image_codes(image).cpu().numpy()
            np.testing.assert_array_equal(codes,sample['sequence_image_codes'][segment,j])
    assert next(interface.vq_model.parameters()).dtype==torch.float32
    assert all(not p.requires_grad for p in interface.vq_model.parameters())
    assert all(p.requires_grad for p in interface.model.parameters())
    if is_oat:
        assert next(model.oat_codec.parameters()).dtype == torch.float32
        assert not model.oat_codec.training
        assert all(not p.requires_grad for p in model.oat_codec.parameters())
        assert not any('oat_codec' in name for name, _ in model.named_parameters())
    # Compare a backbone projection with its original pretrained shard.
    key='model.layers.0.self_attn.q_proj.weight'
    base=Path(cfg.framework.u0.base_vlm)
    index=json.loads((base/'model.safetensors.index.json').read_text())
    with safe_open(base/index['weight_map'][key],framework='pt',device='cpu') as f:
        original=f.get_tensor(key).to(torch.bfloat16)
    current=interface.model.state_dict()[key].detach().cpu()
    changed=int((current!=original).sum())
    assert changed>0
    for file,digest in model.sequence_packer.contract['encoder_sha256'].items():
        with open(file,'rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==digest
    e=dict(image=images[0],state=sample['sequence_state'][0],lang=sample['lang'])
    first=wrapper.predict_action([e],unnorm_key='franka')
    token_key = 'action_tokens' if is_oat else 'fast_tokens'
    history=[dict(**e, **{token_key: first[token_key][0]})]
    second=wrapper.predict_action([dict(image=images[1],state=sample['sequence_state'][1],lang=sample['lang'],history=history)],unnorm_key='franka')
    for result in (first,second):
        assert result['actions'].shape==(1,model.action_horizon,7) and np.isfinite(result['actions']).all()
    report=dict(checkpoint=str(checkpoint),logits_reload_exact=True,online_cache_all_four_views_exact=True,
        backbone_changed_values=changed,encoders_unchanged=True,sequence_tokens=inputs['input_ids'].shape[1],
        first_action_tokens=first[token_key][0],history_action_tokens=second[token_key][0],
        two_action_chunks_valid=True,metadata=wrapper.metadata)
    (root/'gpu_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
