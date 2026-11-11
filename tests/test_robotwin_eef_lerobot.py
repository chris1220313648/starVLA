import json
import random
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from transformers import BatchFeature, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from examples.simBenchmarks.Robotwin.robotwin_eef import eef20, to_ee16, unnormalize
from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_lerobot_dataset import (
    CAMERAS, COMPOSITE_COT_PROMPT, RobotwinEEFLeRobotDataset, ShardCache, map_cot_to_composite, prepare,
)
from starVLA.model.framework.VLM4A import GroundingPIPI05 as pi05

BBOX = '<|object_ref_start|>bottle<|object_ref_end|><|box_start|><371><636><825><917><|box_end|>'
MAPPED_BBOX = '<|object_ref_start|>bottle<|object_ref_end|><|box_start|><371><424><825><611><|box_end|>'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.fixture
def sources(tmp_path):
    root, cot, cache = tmp_path/'eef', tmp_path/'cot', tmp_path/'cache'
    clean = root/'clean'
    names = [f'{arm}_{name}' for arm in ('left', 'right') for name in
             ('x', 'y', 'z', 'r00', 'r10', 'r20', 'r01', 'r11', 'r21', 'gripper')]
    info = dict(codebase_version='v3.0', fps=50, total_episodes=2, total_frames=7,
                data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
                features={k:dict(shape=[20], names=names) for k in ('action', 'observation.state')})
    write_json(clean/'meta/info.json', info)
    source, metadata, frames, mapping, cot_meta, answers, qc = [], [], [], [], [], [], []
    offset = 0
    for eid, n, cid in ((0,4,7),(1,3,3)):
        instruction = tmp_path / 'instructions' / f'episode{eid}.json'
        write_json(instruction, {'seen': [f'task{eid} seen {i}' for i in range(100)], 'unseen': ['do not sample']})
        source.append(dict(episode_index=eid, length=n, source_task=f'task{eid}', local_episode_index=0,
                           task='pick bottle', instruction=str(instruction)))
        ep = dict(episode_index=eid, length=n, dataset_from_index=offset, dataset_to_index=offset+n,
                  **{'data/chunk_index':0,'data/file_index':0})
        for cam in CAMERAS:
            prefix='videos/observation.images.'+cam
            ep.update({prefix+'/chunk_index':0,prefix+'/file_index':0,prefix+'/from_timestamp':(offset+2)/50})
        metadata.append(ep)
        values=np.zeros((n,20),np.float32)
        values[:,0]=np.arange(n)+eid*10
        values[:,[3,7,13,17]]=1
        for f in range(n):
            frames.append(dict(episode_index=eid,frame_index=f,index=offset+f,timestamp=f/50,
                               **{'observation.state':values[f], 'action':values[min(f+1,n-1)]}))
        mapping.append(dict(episode_index=cid,source_task=f'task{eid}',source_episode=0,frames=n-1))
        cot_start=len(answers)
        cot_meta.append(dict(episode_index=cid,length=n-1,dataset_from_index=cot_start,
                             **{'data/chunk_index':0,'data/file_index':0}))
        for f in range(n-1):
            identity=dict(episode_index=cid,frame_index=f,index=cot_start+f)
            available=f!=1
            answers.append(dict(**identity, **{'annotation.cot':BBOX if available else '', 'annotation.cot_available':available}))
            qc.append(dict(**identity, metadata_json=json.dumps({'cot_primary_instruction':'grounded pick bottle'}),
                           target_only_metadata_json=json.dumps({'answer_available':available,'target_coverage_complete':False})))
        offset+=n
    clean.mkdir(exist_ok=True)
    (clean/'source_episodes.jsonl').write_text('\n'.join(json.dumps(s) for s in source))
    parquet(clean/'meta/episodes/chunk-000/file-000.parquet', metadata)
    parquet(clean/'data/chunk-000/file-000.parquet',frames)
    for cam,channel in zip(CAMERAS,range(3)):
        video=clean/f'videos/observation.images.{cam}/chunk-000/file-000.mp4'
        video.parent.mkdir(parents=True)
        with av.open(str(video),'w') as container:
            stream=container.add_stream('libx264',rate=50)
            stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
            for frame in range(9):
                rgb=np.zeros((24,32,3),np.uint8);rgb[:,:,channel]=40+frame*20
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
    dataset=cot/'datasets/groundingpi_target_only'
    write_json(dataset/'meta/info.json',info)
    write_json(dataset/'meta/cot_contract.json',dict(profile='groundingpi_target_only',bbox_camera='observation.images.cam_high',request='Locate targets.'))
    (dataset/'meta/model_visible_setup.txt').write_text('Head image, xyxy 0-999.')
    parquet(dataset/'meta/episodes/chunk-000/file-000.parquet',cot_meta)
    parquet(dataset/'data/chunk-000/file-000.parquet',answers)
    parquet(dataset/'annotations/data/chunk-000/file-000.parquet',qc)
    parquet(cot/'shared/annotations/episode_mapping.parquet',mapping)
    return root,cot,cache


def test_loading_alignment_padding_images_and_optional_cot(sources):
    root,cot,cache=sources
    index=prepare(root,cache,cot)
    assert index['total_samples']==5 and index['cot_episode_mapping']=={0:7,1:3}
    assert prepare(root,cache,cot)['total_samples']==5
    dataset=RobotwinEEFLeRobotDataset(SimpleNamespace(data_root_dir=cache),training=False)
    row=dataset[0]
    np.testing.assert_allclose(unnormalize(row['action'],dataset.stats['robotwin_eef']['action'])[:4,0],[1,2,3,3],atol=1e-5)
    assert row['action_mask'].sum()==3 and dataset[2]['action_mask'].sum()==1
    assert row['cot_episode_index']==7 and dataset[3]['cot_episode_index']==3
    assert row['cot']==MAPPED_BBOX and row['cot_metadata']['target_coverage_complete'] is False
    assert row['lang']=='grounded pick bottle'
    assert dataset[1]['cot'] is None
    assert 'cot_image' not in row and 'cot_prompt' not in row
    assert row['cot_coordinate_frame']=='composite_320x384_v1'
    assert row['image'][0].size==(320,384) and row['state'].shape==(1,20)
    # Frame zero starts at MP4 frame two, not at the beginning of the file.
    assert abs(row['image'][0].getpixel((100,100))[0]-80)<5
    assert row['image'][0].getpixel((100,300))[1]>70
    assert row['image'][0].getpixel((250,300))[2]>70
    plain=cache.parent/'plain'
    prepare(root,plain)
    assert 'cot' not in RobotwinEEFLeRobotDataset(SimpleNamespace(data_root_dir=plain))[0]
    with pytest.raises(IndexError):dataset[len(dataset)]


def test_seen_sampling_survives_cot_attachment_and_is_reproducible(sources, monkeypatch):
    root, cot, cache = sources
    index = prepare(root, cache, cot, instruction_source='seen')
    assert all(len(ep['seen_instructions']) == 100 for ep in index['episodes'])
    cfg = SimpleNamespace(data_root_dir=cache, instruction_source='seen')
    dataset = RobotwinEEFLeRobotDataset(cfg)
    with monkeypatch.context() as patch:
        patch.setattr(random, 'choice', lambda texts: texts[-1])
        assert dataset[0]['lang'] == 'task0 seen 99'
        assert dataset[1]['cot'] is None and dataset[1]['lang'] == 'task0 seen 99'
        assert dataset[3]['lang'] == 'task1 seen 99'
    state = random.getstate()
    try:
        random.seed(42)
        first = [dataset[0]['lang'] for _ in range(8)]
        random.seed(42)
        assert first == [dataset[0]['lang'] for _ in range(8)]
        assert len(set(first)) > 1
    finally:
        random.setstate(state)
    assert RobotwinEEFLeRobotDataset(cfg, training=False)[0]['lang'] == 'task0 seen 0'
    with pytest.raises(ValueError, match='instruction_source disagree'):
        RobotwinEEFLeRobotDataset(SimpleNamespace(data_root_dir=cache))


def test_seen_cache_tracks_instruction_content_and_mode(sources):
    root, cot, cache = sources
    prepare(root, cache, cot, instruction_source='seen')
    assert prepare(root, cache, cot, instruction_source='seen')['instruction_source'] == 'seen'
    with pytest.raises(ValueError, match='source changed'):
        prepare(root, cache, cot)
    source = json.loads((root/'clean/source_episodes.jsonl').read_text().splitlines()[0])
    path = Path(source['instruction'])
    texts = json.loads(path.read_text())
    texts['seen'][0] = 'changed instruction'
    write_json(path, texts)
    with pytest.raises(ValueError, match='source changed'):
        prepare(root, cache, cot, instruction_source='seen')


@pytest.mark.parametrize('texts', [[], ['x']*99, ['x']*99+[''], ['x']*99+[None]])
def test_seen_rejects_invalid_instruction_pools(sources, texts):
    root, cot, cache = sources
    source = json.loads((root/'clean/source_episodes.jsonl').read_text().splitlines()[0])
    write_json(Path(source['instruction']), {'seen': texts})
    with pytest.raises(ValueError, match='100 valid seen instructions'):
        prepare(root, cache, cot, instruction_source='seen')


def test_separate_views_keep_head_camera_cot_coordinates(sources):
    root,cot,cache=sources
    separate_cache=cache.parent/'separate'
    index=prepare(root,separate_cache,cot,image_input_mode='separate')
    assert index['image_hw']==[240,320]
    row=RobotwinEEFLeRobotDataset(SimpleNamespace(data_root_dir=separate_cache,
        image_input_mode='separate'),training=False)[0]
    assert len(row['image'])==3
    assert all(image.size==(320,240) for image in row['image'])
    assert row['cot']==BBOX
    assert row['cot_coordinate_frame']=='head_camera_320x240_v1'


@pytest.mark.parametrize('single_row', [False, True])
def test_rgb_transition_tables_keep_last_sample_and_next_pose(sources, single_row):
    root,cot,cache=sources
    clean=root/'clean'
    source=[json.loads(s) for s in (clean/'source_episodes.jsonl').read_text().splitlines()]
    metadata=pd.read_parquet(clean/'meta/episodes/chunk-000/file-000.parquet')
    frames=pd.read_parquet(clean/'data/chunk-000/file-000.parquet')
    kept=[];offset=0
    for i,s in enumerate(source):
        n=s['length']-1 if not (single_row and i==1) else 1
        s.update(length=n,raw_length=n+1)
        rows=frames[frames.episode_index==i].iloc[:n].copy()
        rows['index']=np.arange(offset,offset+n)
        kept.append(rows)
        metadata.loc[i,['length','dataset_from_index','dataset_to_index']]=[n,offset,offset+n]
        offset+=n
    (clean/'source_episodes.jsonl').write_text('\n'.join(json.dumps(s) for s in source))
    pd.concat(kept).to_parquet(clean/'data/chunk-000/file-000.parquet',index=False)
    metadata.to_parquet(clean/'meta/episodes/chunk-000/file-000.parquet',index=False)
    info=json.loads((clean/'meta/info.json').read_text());info['total_frames']=offset
    for feature in info['features'].values():feature['names']=[feature['names']]
    write_json(clean/'meta/info.json',info)
    if single_row:
        dataset=cot/'datasets/groundingpi_target_only'
        for base in [dataset/'data',dataset/'annotations/data']:
            path=base/'chunk-000/file-000.parquet';rows=pd.read_parquet(path)
            rows[~((rows.episode_index==3)&(rows.frame_index==1))].to_parquet(path,index=False)
        path=dataset/'meta/episodes/chunk-000/file-000.parquet';rows=pd.read_parquet(path)
        rows.loc[rows.episode_index==3,'length']=1;rows.to_parquet(path,index=False)
        path=cot/'shared/annotations/episode_mapping.parquet';rows=pd.read_parquet(path)
        rows.loc[rows.episode_index==3,'frames']=1;rows.to_parquet(path,index=False)
    index=prepare(root,cache,cot)
    assert index['total_samples']==offset==index['total_frames']
    dataset=RobotwinEEFLeRobotDataset(SimpleNamespace(data_root_dir=cache),training=False)
    assert len(dataset)==offset
    last=dataset[len(dataset)-1]
    action=unnormalize(last['action'],dataset.stats['robotwin_eef']['action'])
    state=unnormalize(last['state'],dataset.stats['robotwin_eef']['state'])
    assert last['action_mask'].sum()==1 and last['cot_episode_index']==3
    np.testing.assert_allclose(action[:,0],11 if single_row else 12,atol=1e-5)
    np.testing.assert_allclose(action[0,0]-state[0,0],1,atol=1e-5)
    assert not dataset._static(dataset._window(len(dataset)-1))


def test_cached_scene_manifest_rejects_changed_missing_and_wrong_revision(tmp_path):
    from examples.simBenchmarks.Robotwin.eval_files.evaluate_groundingpi_cached import digest, verify_cache
    tasks=['task_a','task_b'];hashes={}
    write_json(tmp_path/'provenance.json',{'revisions':{'robotwin':'fixed'}})
    for suite in ['demo_clean','demo_randomized']:
        for task in tasks:
            path=tmp_path/f'seed-cache/{suite}/{task}.pkl';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'cached scene');hashes[str(path.relative_to(tmp_path))]=digest(path)
    write_json(tmp_path/'seed-hashes.json',hashes)
    assert verify_cache(tmp_path,tasks,'fixed')==hashes
    with pytest.raises(ValueError,match='different RoboTwin revision'):verify_cache(tmp_path,tasks,'other')
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed scene cache'):verify_cache(tmp_path,tasks,'fixed')
    path.unlink()
    with pytest.raises(ValueError,match='Missing'):verify_cache(tmp_path,tasks,'fixed')


def test_worker_sweep_uses_wall_time_and_confirmed_tie_break():
    from examples.simBenchmarks.Robotwin.train_files.benchmark_groundingpi_workers import measured_seconds, choose_workers
    log='50/200 [02:00<05:00, 2.0s/it]\r50/200 [02:00<05:00, 2.0s/it]\n190/200 [06:40<00:20, 2.0s/it]'
    assert measured_seconds(log)==2
    with pytest.raises(ValueError,match='boundaries'):measured_seconds('50/200 [00:20<01:00]')
    rows=[dict(workers=2,seconds_per_step=4),dict(workers=4,seconds_per_step=2.02),
          dict(workers=8,seconds_per_step=2),dict(workers=16,seconds_per_step=3),
          dict(workers=4,seconds_per_step=2.04),dict(workers=8,seconds_per_step=2)]
    selected,averages=choose_workers(rows)
    assert selected==4 and averages==pytest.approx({4:2.03,8:2})
    rows[-1]['seconds_per_step']=1
    assert choose_workers(rows)[0]==8


def test_source_mapping_rejects_missing_duplicate_and_wrong_lengths(sources):
    root,cot,cache=sources
    mapping=cot/'shared/annotations/episode_mapping.parquet'
    rows=pd.read_parquet(mapping)
    rows.loc[0,'frames']=99;rows.to_parquet(mapping,index=False)
    with pytest.raises(ValueError,match='length mismatch'):prepare(root,cache,cot)
    rows.loc[0,'frames']=3
    pd.concat([rows,rows.iloc[:1]]).to_parquet(mapping,index=False)
    with pytest.raises(ValueError,match='Duplicate CoT source'):prepare(root,cache,cot)


def test_shard_cache_bound_and_provenance(sources):
    root,cot,cache=sources
    prepare(root,cache)
    with pytest.raises(ValueError,match='source changed'):prepare(root,cache,cot)
    tables=ShardCache()
    for i in range(3):
        p=cache/f'{i}.parquet';parquet(p,[dict(episode_index=i,frame_index=0)])
        tables.read(p)
    assert len(tables.tables)==2 and cache/'0.parquet' not in tables.tables
    with pytest.raises(ValueError,match='separate'):prepare(root,root/'cache')


def test_quaternion_orders():
    from scipy.spatial.transform import Rotation
    q=Rotation.from_euler('xyz',[.2,-.7,.3]).as_quat()
    xyzw=np.r_[1,2,3,q];wxyz=np.r_[1,2,3,q[[3,0,1,2]]]
    old=eef20(xyzw,xyzw,.2,.8)
    new=eef20(wxyz,wxyz,.2,.8,quaternion_order='wxyz')
    np.testing.assert_allclose(old,new,atol=1e-6)
    recovered=to_ee16(new,quaternion_order='wxyz')
    np.testing.assert_allclose(Rotation.from_quat(recovered[3:7][[1,2,3,0]]).as_matrix(),Rotation.from_quat(q).as_matrix(),atol=1e-6)
    identity=np.array([0,0,0,1,0,0,0])
    np.testing.assert_allclose(eef20(identity,identity,0,0,quaternion_order='wxyz')[3:9],[1,0,0,0,1,0])


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model=Qwen3Model(Qwen3Config(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,head_dim=8,vocab_size=16,attention_dropout=0))
        self.visual=nn.Linear(3,32,bias=False)

    def get_input_embeddings(self):return self.language_model.embed_tokens

    def get_image_features(self,pixels,grid):return list(self.visual(pixels).unsqueeze(1).unbind(0))

    def get_placeholder_mask(self,ids,embeddings,image_features):
        return ids.eq(1).unsqueeze(-1).expand_as(embeddings),None


class Interface(nn.Module):
    def __init__(self,config):
        super().__init__();self.model=nn.Module();self.model.model=Backbone();self.model.lm_head=nn.Linear(32,16,bias=False)

    def build_qwenvl_inputs(self,images,prompts,solutions=None,allow_missing_solutions=False):
        sequences,labels,prefixes=[],[],[]
        for i,prompt in enumerate(prompts):
            prefix=[1,2,3,4+sum(map(ord,prompt))%11]
            solution=solutions[i] if solutions is not None else None
            answer=[5+ord(c)%10 for c in solution]+[4] if solution is not None else []
            sequences.append(prefix+answer)
            labels.append([-100]*len(prefix)+answer)
            prefixes.append([True]*len(prefix)+[False]*len(answer))
        length=max(map(len,sequences))
        ids=torch.tensor([[0]*(length-len(row))+row for row in sequences])
        mask=ids.ne(0)
        result=dict(input_ids=ids,attention_mask=mask,
                    pixel_values=torch.tensor(np.stack([np.asarray(views[0]).mean((0,1))/255 for views in images]),dtype=torch.float32),
                    image_grid_thw=torch.ones(len(images),3,dtype=torch.long))
        if solutions is not None:
            result.update(labels=torch.tensor([[-100]*(length-len(row))+row for row in labels]),
                          prefix_mask=torch.tensor([[False]*(length-len(row))+row for row in prefixes]))
        return BatchFeature(result)


@pytest.fixture
def cot_model(monkeypatch):
    torch.manual_seed(42)
    monkeypatch.setattr(pi05,'GroundingPIInterface',Interface)
    return pi05.GroundingPIPI05(OmegaConf.create({'framework':{'cot_loss_weight':1.,
        'cot_training_mode':'shared_canvas_v1','cot_prompt':COMPOSITE_COT_PROMPT,'action_model':{
        'action_horizon':3,'action_dim':2,'state_dim':3,'expert_width':16,'expert_mlp_dim':32,
        'state_representation':'continuous_expert_token'}}}))


def sample(answer=True):
    return dict(image=[Image.new('RGB',(320,384),(32,64,96))],cot=MAPPED_BBOX if answer else None,
                cot_coordinate_frame='composite_320x384_v1',lang='pick bottle',state=np.zeros((1,3),np.float32),
                action=np.zeros((3,2),np.float32),action_mask=np.ones(3,dtype=bool))


def shared_output(model,examples):
    return model(examples,noise=torch.ones(len(examples),3,2),time=torch.full((len(examples),),.5))


def test_bbox_mapping_and_validation():
    assert map_cot_to_composite(BBOX)==MAPPED_BBOX
    multi=BBOX.replace('<825><917><|box_end|>','<825><917>,<0><3><999><999><|box_end|>')
    assert map_cot_to_composite(multi)==MAPPED_BBOX.replace('<825><611><|box_end|>','<825><611>,<0><2><999><666><|box_end|>')
    boundary=BBOX.replace('<371><636><825><917>','<0><0><999><999>')
    assert '<0><0><999><666>' in map_cot_to_composite(boundary)
    assert map_cot_to_composite(BBOX+'\n'+boundary)==MAPPED_BBOX+'\n'+map_cot_to_composite(boundary)
    assert map_cot_to_composite(BBOX+', '+boundary)==MAPPED_BBOX+', '+map_cot_to_composite(boundary)
    rounding=BBOX.replace('<371><636><825><917>','<1><1><2><2>')
    assert '<1><1><2><1>' in map_cot_to_composite(rounding)
    for bad in ('', 'bottle bbox', BBOX+'junk', BBOX.replace('<825>','<1000>'), BBOX.replace('<825>','<1>'), ','+BBOX, BBOX+',', BBOX+',,'+BBOX):
        with pytest.raises(ValueError):map_cot_to_composite(bad)


def test_single_visual_and_language_pass(cot_model):
    model=cot_model.eval()
    model.cot_loss_weight = .1
    calls=[0]*(1+len(model.language_model.layers))
    def count(i):
        def hook(*args):calls[i]+=1
        return hook
    handles=[model.qwen_vl_interface.model.model.visual.register_forward_hook(count(0))]
    handles.extend(layer.self_attn.k_proj.register_forward_hook(count(i+1)) for i,layer in enumerate(model.language_model.layers))
    with torch.no_grad():output=shared_output(model,[sample(),sample(False)])
    for handle in handles:handle.remove()
    assert calls==[1,1,1]
    assert output['cot_samples']==1 and torch.isfinite(output['total_loss'])
    torch.testing.assert_close(output['total_loss'], output['action_loss'] + .1*output['cot_loss'])


def test_joint_gradients_and_empty_batches(cot_model):
    model=cot_model.train()
    output=shared_output(model,[sample(),sample(False)])
    output['action_loss'].backward()
    assert model.language_model.layers[0].self_attn.k_proj.weight.grad.abs().sum()>0
    assert model.action_model.action_out_proj.weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True)
    output=shared_output(model,[sample(),sample(False)])
    output['cot_loss'].backward()
    assert model.qwen_vl_interface.model.lm_head.weight.grad.abs().sum()>0
    assert model.language_model.layers[-1].mlp.down_proj.weight.grad.abs().sum()>0
    assert model.qwen_vl_interface.model.model.visual.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.action_model.parameters())
    model.zero_grad(set_to_none=True)
    output=shared_output(model,[sample(False)])
    assert output['cot_loss']==0 and output['cot_samples']==0
    output['cot_loss'].backward()
    assert model.qwen_vl_interface.model.lm_head.weight.grad is not None
    assert model.qwen_vl_interface.model.lm_head.weight.grad.abs().sum()==0
    assert model.language_model.norm.weight.grad is not None
    assert torch.isfinite(output['action_loss'])


def test_assistant_loss_masks_prefix(cot_model):
    hidden=torch.randn(1,5,32,requires_grad=True)
    labels=torch.tensor([[-100,-100,-100,3,4]])
    loss=cot_model._fast_ce(hidden,labels,normalized=True)
    loss.backward()
    assert hidden.grad[:,:2].abs().sum()==0
    assert hidden.grad[:,2:4].abs().sum()>0
    assert hidden.grad[:,-1].abs().sum()==0


def test_cot_ddp_token_normalization(cot_model,monkeypatch):
    hidden=torch.randn(1,4,32)
    labels=torch.tensor([[-100,-100,3,4]])
    reference,_=cot_model._cot_loss(hidden,labels,[sample()])
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:2)
    calls=iter([2,1,None])
    def reduce(value):
        addition=next(calls)
        value.add_(addition) if addition is not None else value.mul_(2)
    monkeypatch.setattr(torch.distributed,'all_reduce',reduce)
    loss,count=cot_model._cot_loss(hidden,labels,[sample()])
    torch.testing.assert_close(loss,reference)
    assert count==2


def test_answers_never_condition_actions_and_cache_matches(cot_model):
    model=cot_model.eval()
    row=sample()
    noisy,time=torch.randn(1,3,2),torch.tensor([.5])
    with torch.no_grad():
        first,_,_=model._cot_velocity([row],noisy,time)
        altered={**row,'cot':MAPPED_BBOX+'\n'+MAPPED_BBOX}
        second,_,_=model._cot_velocity([altered],noisy,time)
        missing,_,_=model._cot_velocity([{**row,'cot':None}],noisy,time)
        prefix,mask=model.prepare_prefix([row])
        full=model.velocity(prefix,mask,noisy,time,state=model._continuous_state([row]))
        cached=model.velocity(None,mask,noisy,time,cache=model.prefix_cache(prefix,mask),state=model._continuous_state([row]))
    for actual in (second,missing,full,cached):torch.testing.assert_close(first,actual,atol=1e-6,rtol=1e-5)
    labelled=model._prefix_inputs([row],solutions=[row['cot']])
    inference=model._prefix_inputs([row])
    torch.testing.assert_close(labelled.input_ids[labelled.prefix_mask],inference.input_ids[inference.attention_mask])
    assert 'ENTIRE composite' in model._prefix_prompts([row])[0]


def test_text_causality_and_prefix_isolation(cot_model):
    model=cot_model.eval()
    hidden=torch.randn(1,5,32)
    labels=torch.tensor([[-100,-100,3,4,5]])
    mask=torch.ones(1,5,dtype=torch.bool)
    attention,prefix=model.ki_attention(mask,labels)
    changed=hidden.clone();changed[:,-1]+=100
    positions=torch.arange(5)[None]
    with torch.no_grad():
        original,*_=model._ki_layer(0,hidden,attention,positions)
        altered,*_=model._ki_layer(0,changed,attention,positions)
    torch.testing.assert_close(original[:,:4],altered[:,:4],atol=0,rtol=0)
    assert prefix.sum()==2 and not attention[0,0,:2,2:].any()


def test_shared_checkpoint_contract_and_legacy_inference(cot_model):
    model=cot_model.eval()
    restored=pi05.GroundingPIPI05(model.config).eval()
    restored.load_state_dict(model.state_dict(),strict=True)
    noise=np.ones((1,3,2),np.float32)
    np.testing.assert_array_equal(model.predict_action([sample()],noise=noise,num_inference_steps=2)['normalized_actions'],
                                  restored.predict_action([sample()],noise=noise,num_inference_steps=2)['normalized_actions'])
    from starVLA.training.trainer_utils.config_tracker import wrap_config
    tracked=wrap_config(OmegaConf.create(OmegaConf.to_container(model.config)))
    pi05.GroundingPIPI05(tracked)
    exported=tracked.export_accessed_config()['framework']
    assert exported['cot_training_mode']=='shared_canvas_v1' and exported['cot_prompt']==COMPOSITE_COT_PROMPT
    legacy=OmegaConf.create(OmegaConf.to_container(model.config))
    del legacy.framework.cot_training_mode
    del legacy.framework.cot_prompt
    old=pi05.GroundingPIPI05(legacy).eval()
    old.load_state_dict(model.state_dict(),strict=True)
    assert old._prefix_prompts([sample()])==['Task: pick bottle;\nAction: ']
    assert np.isfinite(old.predict_action([sample()],noise=noise,num_inference_steps=2)['normalized_actions']).all()


def _ddp_cot_worker(rank,config,rendezvous):
    pi05.GroundingPIInterface=Interface
    model=pi05.GroundingPIPI05(OmegaConf.create(config))
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        wrapped=torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
        for iteration in range(2):
            wrapped.zero_grad(set_to_none=True)
            output=shared_output(wrapped,[sample(answer=rank==0 and iteration==0)])
            assert output['cot_samples']==(1 if iteration==0 else 0) and torch.isfinite(output['total_loss'])
            output['total_loss'].backward()
            grad=model.qwen_vl_interface.model.lm_head.weight.grad
            assert grad is not None
            assert (grad.abs().sum()>0) if iteration==0 else (grad.abs().sum()==0)
            other=grad.clone();torch.distributed.broadcast(other,src=0)
            torch.testing.assert_close(other,grad)
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_training_with_missing_answers(cot_model,tmp_path):
    torch.multiprocessing.start_processes(_ddp_cot_worker,args=(OmegaConf.to_container(cot_model.config),str(tmp_path/'gloo')),
                                        nprocs=2,join=True,start_method='spawn')


@pytest.mark.parametrize('instruction_source', ['default', 'seen'])
def test_lerobot_launch_preserves_dataset_and_cot_flags(sources, instruction_source):
    from examples.simBenchmarks.Robotwin.train_files import groundingpi_robotwin as runner
    root,cot,cache=sources
    args=SimpleNamespace(framework='pi05_eef',dataset_root=root,cot_root=cot,with_cot=True,cot_loss_weight=.1,
                         instruction_source=instruction_source,
                         data_root=cache,raw_data=cache.parent/'unused_hdf5',model=cache.parent/'model',
                         output=cache.parent/'run',batch_size=2,workers=0,steps=3,gpus=1,train_port=29615,
                         image_size=224,dry_run=True)
    manifest=runner.prepare(args)
    write_json(runner.validation_path(args),dict(passed=True,manifest=manifest,
        model_contract={'image_size':[384,320],'image_count':1}))
    write_json(args.model/'config.json',{})
    runner.train(args)
    cfg=OmegaConf.load(args.output/'launch.yaml')
    assert cfg.datasets.vla_data.dataset_py=='robotwin_eef_lerobot'
    assert cfg.datasets.vla_data.quaternion_order=='wxyz'
    assert 'raw_data_dir' not in cfg.datasets.vla_data
    assert cfg.framework.cot_loss_weight==.1
    assert cfg.datasets.vla_data.instruction_source==instruction_source
    assert cfg.framework.cot_training_mode=='shared_canvas_v1' and cfg.framework.cot_prompt==COMPOSITE_COT_PROMPT
    command=(args.output/'launch_command.sh').read_text()
    for flag in ('--dataset-root','--with-cot','--cot-root','--cot-loss-weight','--instruction-source'):
        assert flag in command
    assert f'--instruction-source {instruction_source}' in command


def test_new_client_uses_server_wxyz_metadata(monkeypatch):
    from examples.simBenchmarks.Robotwin.eval_files import model2robotwin_interface as client
    requests=[]
    pose=np.array([.1,.2,.3,1,0,0,0])
    action=eef20(pose,pose,.25,.75,quaternion_order='wxyz')
    class Connection:
        def __init__(self,*args):pass
        def get_server_metadata(self):return {'action_chunk_size':32,'quaternion_order':'wxyz'}
        def predict_action(self,request):
            requests.append(request)
            return {'data':{'actions':np.tile(action,(1,32,1))}}
    monkeypatch.setattr(client,'WebsocketClientPolicy',Connection)
    model=client.get_model(dict(policy_ckpt_path='',include_state=True,action_type='ee'))
    executed=[]
    class Environment:
        take_action_cnt=0
        def get_instruction(self):return 'pick'
        def take_action(self,action,**kwargs):executed.append(action)
    observation=dict(observation={key:{'rgb':np.zeros((24,32,3),np.uint8)} for key in ('head_camera','left_camera','right_camera')},
                     endpose=dict(left_endpose=pose,right_endpose=pose,left_gripper=.25,right_gripper=.75))
    client.eval(Environment(),model,observation)
    np.testing.assert_allclose(requests[0]['examples'][0]['state'],action[None])
    np.testing.assert_allclose(executed[0][3:7],pose[3:7])


@pytest.mark.parametrize('frame,expected', [
    (0, [0,0,0,0,0]), (39, [0,39,39,39,39]), (40, [0,40,40,0,40]),
    (50, [0,50,50,10,50]), (80, [0,80,0,40,80]),
    (119, [0,119,39,79,119]), (120, [0,0,40,80,120]),
    (160, [0,40,80,120,160]),
])
def test_history_frame_indices(frame, expected):
    from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_lerobot_dataset import image_frame_indices
    assert image_frame_indices(frame, 4, 40) == expected
    assert image_frame_indices(frame) == [frame]


def test_history_loading_and_shared_color_jitter(sources, monkeypatch):
    root, cot, cache = sources
    prepare(root, cache, cot)
    cfg = SimpleNamespace(data_root_dir=cache, history_frames=4, history_stride=40, color_jitter=.2)
    plain = RobotwinEEFLeRobotDataset(cfg, training=False)
    augmented = RobotwinEEFLeRobotDataset(cfg, training=True)
    calls = []
    def uniform(low, high):
        calls.append((low, high))
        return 1.15
    monkeypatch.setattr(random, 'uniform', uniform)
    reference = plain[3]
    assert calls == []
    row = augmented[3]  # The second episode must start with its own first frame.
    assert len(calls) == 3 and calls == [(.8, 1.2)] * 3
    assert len(row['image']) == 5 and row['episode_index'] == reference['episode_index']
    assert all(image.mode == 'RGB' and image.size == (320,384) for image in row['image'])
    assert all(np.array_equal(row['image'][0], image) for image in row['image'])
    assert not np.array_equal(row['image'][0], reference['image'][0])
    assert row['cot'] == reference['cot']
    np.testing.assert_array_equal(row['state'], reference['state'])
    np.testing.assert_array_equal(row['action'], reference['action'])


def test_history_cot_prefix_and_backward(cot_model):
    cot_model.cot_training_mode = 'history_canvas_v1'
    row = sample()
    row['image'] *= 5
    output = shared_output(cot_model, [row])
    (output['action_loss'] + output['cot_loss']).backward()
    assert torch.isfinite(output['cot_loss'])
    row['image'] = row['image'][:4]
    with pytest.raises(ValueError, match='5 320x384'):
        cot_model.prepare_prefix([row])


def test_shared_annotation_delivery_layout(sources):
    import shutil
    root, cot, cache = sources
    shutil.move(str(cot/'datasets/groundingpi_target_only/annotations/data'), str(cot/'shared/annotations/data'))
    (cot/'datasets/groundingpi_target_only/annotations').rmdir()
    prepare(root, cache, cot)
    dataset = RobotwinEEFLeRobotDataset(SimpleNamespace(data_root_dir=cache), training=False)
    assert dataset[0]['cot'] == MAPPED_BBOX


def test_history_decoder_timestamps_and_order(sources, monkeypatch):
    from starVLA.dataloader.gr00t_lerobot import video
    root, cot, cache = sources
    prepare(root, cache)
    dataset = RobotwinEEFLeRobotDataset(SimpleNamespace(data_root_dir=cache, history_frames=4, history_stride=40), training=False)
    ep, _, rows = dataset._window(0)
    ep = dict(ep, length=201, sample_length=201)
    rows = pd.concat([rows.iloc[:1]] * 201, ignore_index=True)
    monkeypatch.setattr(dataset, '_window', lambda index: (ep, 160, rows))
    requested = []
    def decode(path, timestamps, video_backend):
        start = next(value['start'] for value in ep['videos'].values() if value['path'] == path)
        indices = [round((time-start)*50) for time in timestamps]
        requested.append(indices)
        return np.stack([np.full((24,32,3), step, np.uint8) for step in indices])
    monkeypatch.setattr(video, 'get_frames_by_timestamps', decode)
    row = dataset[0]
    assert requested == [[0,40,80,120,160]] * 3
    assert [im.getpixel((100,100))[0] for im in row['image']] == [0,40,80,120,160]
    assert row['frame_index'] == 160
