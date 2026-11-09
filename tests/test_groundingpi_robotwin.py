"""RoboTwin Clean v3 data/normalization and launch regression checks."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'examples/simBenchmarks/Robotwin/train_files/groundingpi_robotwin.py'
spec = importlib.util.spec_from_file_location('groundingpi_robotwin_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def data(tmp_path):
    raw, adapter = tmp_path/'raw', tmp_path/'adapter'
    source = raw/'clean'
    (source/'meta/episodes/chunk-000').mkdir(parents=True)
    (source/'data/chunk-000').mkdir(parents=True)
    features = {key: {'shape':[14]} for key in ('action','observation.state')}
    features.update({'observation.images.'+cam: {'shape':[2,3,3], 'names':['height','width','rgb'],
        'info':{'video.channels':3,'video.fps':50}} for cam in runner.CAMERAS})
    info = dict(codebase_version='v3.0',fps=50,total_episodes=2,total_frames=6,
                total_videos=6,chunks_size=1000,features=features,
                data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
    (source/'meta/info.json').write_text(json.dumps(info))
    pd.DataFrame({'task_index':[7,3],'task':['second instruction','first instruction']}).to_parquet(source/'meta/tasks.parquet',index=False)
    episodes=[]
    rows=[]
    for ep_id in range(2):
        ep={'episode_index':ep_id,'length':3,'tasks':['instruction'],'data/chunk_index':0,'data/file_index':0,
            'dataset_from_index':ep_id*3,'dataset_to_index':ep_id*3+3}
        for i,cam in enumerate(runner.CAMERAS):
            prefix='videos/observation.images.'+cam
            ep.update({prefix+'/chunk_index':7+i,prefix+'/file_index':9+i,
                       prefix+'/from_timestamp':ep_id*20.,prefix+'/to_timestamp':ep_id*20.+.06})
            p=source/info['video_path'].format(video_key='observation.images.'+cam,chunk_index=7+i,file_index=9+i)
            p.parent.mkdir(parents=True,exist_ok=True);p.touch()
        episodes.append(ep)
        for step in range(3):
            values=np.arange(14,dtype=np.float32)/20 + ep_id*.1+step*.01
            values[[6,13]]=[0,1]
            rows.append({'episode_index':ep_id,'frame_index':step,'index':ep_id*3+step,
                         'timestamp':step/50,'task_index':3 if ep_id==0 else 7,
                         'observation.state':values,'action':values})
    pd.DataFrame(episodes).to_parquet(source/'meta/episodes/chunk-000/file-000.parquet',index=False)
    pd.DataFrame(rows).to_parquet(source/'data/chunk-000/file-000.parquet',index=False)
    args=argparse.Namespace(raw_data=raw,data_root=adapter,output=tmp_path/'output',
        model=tmp_path/'model',batch_size=16,workers=0,steps=10000,gpus=8,
        train_port=29527,sample_episodes=2,dry_run=True)
    return args


def test_prepare_preserves_source_and_rejects_contract_changes(data):
    source=data.raw_data/'clean'
    before={str(p.relative_to(source)):p.read_bytes() for p in (source/'meta').rglob('*') if p.is_file()}
    manifest=runner.prepare(data)
    (data.data_root/'clean/meta/stats_gr00t.json').write_text('local cache')
    assert runner.prepare(data)==manifest
    assert {str(p.relative_to(source)):p.read_bytes() for p in (source/'meta').rglob('*') if p.is_file()}==before
    assert (data.data_root/'clean/videos').resolve()==source/'videos'
    assert not (data.data_root/'clean/meta').is_symlink()
    assert not (source/'meta/modality.json').exists()
    saved=json.loads((data.data_root/'clean/meta/modality.json').read_text())
    assert list(saved['action'])==['left_joints','right_joints','left_gripper','right_gripper']
    assert saved['action']['right_joints']['start']==7
    (source/'meta/info.json').write_text((source/'meta/info.json').read_text()+' ')
    with pytest.raises(ValueError,match='changed'):
        runner.prepare(data)
    data.data_root=data.raw_data
    with pytest.raises(ValueError,match='separate'):
        runner.prepare(data)


@pytest.mark.parametrize('layout',['columns','text_index','id_index'])
def test_v3_task_layout_and_noncontiguous_ids(tmp_path,layout):
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
    root=tmp_path; (root/'meta').mkdir()
    frame=pd.DataFrame({'task_index':[7,3],'task':['pick bottle','place cup']})
    if layout=='text_index':frame=frame.set_index('task')
    if layout=='id_index':frame=frame.set_index('task_index')
    frame.to_parquet(root/'meta/tasks.parquet')
    ds=LeRobotSingleDataset.__new__(LeRobotSingleDataset)
    ds._dataset_path=root;ds._lerobot_version='v3.0'
    tasks=ds._get_tasks()
    assert tasks.loc[[3,7]]['task'].tolist()==['place cup','pick bottle']
    assert list(tasks.columns)==['task']


def test_frame_checks_detect_wrong_fps_and_missing_shards(data):
    runner.prepare(data)
    episodes,boundaries,counts=runner.check_tables(data.raw_data/'clean')
    assert counts['episodes']==2 and counts['frames']==6 and boundaries=={0,1}
    p=next((data.raw_data/'clean/data').glob('*/*.parquet'))
    frame=pd.read_parquet(p);frame.loc[1,'timestamp']=1/15;frame.to_parquet(p,index=False)
    with pytest.raises(AssertionError):runner.check_tables(data.raw_data/'clean')


def test_real_loader_camera_offsets_order_padding_and_normalization(data,monkeypatch):
    import starVLA.dataloader.gr00t_lerobot.datasets as datasets
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from deployment.model_server import policy_norm_processor as norm
    runner.prepare(data)
    calls=[]
    def decode(path,timestamps,**kwargs):
        calls.append((path,np.asarray(timestamps).copy()))
        cam=next(i for i,c in enumerate(runner.CAMERAS) if c in path)
        return np.full((len(timestamps),2,3,3),30+cam*40,np.uint8)
    monkeypatch.setattr(datasets,'get_frames_by_timestamps',decode)
    cfg=runner.config(data)
    mixture=get_vla_dataset(cfg.datasets.vla_data)
    subset=mixture.datasets[0]
    assert subset.dataset_name=='clean' and len(mixture.datasets)==1
    source=subset.get_trajectory_data(1)
    raw=subset.get_step_data(1,2)
    assert raw[subset.modality_keys['language'][0]]==['second instruction']
    for i,(cam,(path,timestamps)) in enumerate(zip(runner.CAMERAS,calls)):
        expected_path=subset.dataset_path/subset.video_path_pattern.format(
            video_key='observation.images.'+cam,chunk_index=7+i,file_index=9+i)
        assert Path(path)==expected_path
        np.testing.assert_allclose(timestamps,[20.04])
    packed_raw=np.concatenate([raw[k] for k in subset.modality_keys['action']],axis=1)
    expected=np.stack(source['action'])[-1,runner.ORDER]
    np.testing.assert_array_equal(packed_raw,np.tile(expected,(16,1)))
    transformed=subset.transforms(raw)
    sample=subset._pack_sample(transformed)
    assert sample['action'].shape==(16,14) and sample['state'].shape==(1,14)
    assert [np.asarray(im)[0,0,0] for im in sample['image']]==[30,70,110]
    stats=data.data_root/'dataset_statistics.json';mixture.save_dataset_statistics(stats)
    saved=json.loads(stats.read_text())
    monkeypatch.setattr(norm,'read_mode_config',lambda _: (OmegaConf.to_container(cfg,resolve=True),saved))
    processor=norm.PolicyNormProcessor('unused')
    np.testing.assert_allclose(processor.unapply_actions(sample['action']),np.tile(expected,(16,1)),atol=1e-6)
    state=np.stack(source['observation.state'])[-1:,runner.ORDER]
    np.testing.assert_allclose(processor.apply_state(state),sample['state'],atol=1e-6)


@pytest.mark.parametrize('size,accum', [(224,1),(448,2)])
def test_dry_run_requires_validation_and_never_launches(data,monkeypatch,size,accum):
    data.image_size=size
    data.gradient_accumulation_steps=accum
    manifest=runner.prepare(data)
    with pytest.raises(FileNotFoundError):runner.train(data)
    runner.write_json(runner.validation_path(data),{'passed':False,'manifest':manifest})
    with pytest.raises(ValueError,match='validate'):runner.train(data)
    runner.write_json(runner.validation_path(data),{'passed':True,'manifest':manifest,
        'model_contract':{'image_size':[size,size]}})
    runner.write_json(data.data_root/'dataset_statistics.json',{})
    data.model.mkdir();(data.model/'config.json').write_text('{}')
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:pytest.fail('dry run launched training'))
    runner.train(data)
    cfg=OmegaConf.load(data.output/'launch.yaml')
    assert cfg.framework.action_model.action_dim==14
    assert cfg.framework.action_model.action_horizon==16
    assert cfg.framework.action_model.state_dim==0
    assert cfg.datasets.vla_data.include_state and cfg.datasets.vla_data.action_mode=='abs'
    assert cfg.trainer.max_train_steps==10000
    assert list(cfg.datasets.vla_data.image_size)==[size,size]
    assert list(cfg.datasets.vla_data.obs_image_size)==[size,size]
    assert cfg.trainer.gradient_accumulation_steps==accum
    assert json.loads((data.output/'deepspeed.json').read_text())['gradient_accumulation_steps']==accum
    command = (data.output/'launch_command.sh').read_text()
    assert 'groundingpi_robotwin.py train' in command and '--dry-run' not in command
    assert f'--image-size {size}' in command and f'--gradient-accumulation-steps {accum}' in command
    (data.output/'config.full.yaml').touch()
    with pytest.raises(FileExistsError):runner.train(data)


def test_v3_forward_accepts_three_views_state_tokens_and_14_actions(monkeypatch):
    import torch
    from torch import nn
    from PIL import Image
    import starVLA.model.framework.VLM4A.QwenPI_v3 as v3
    from starVLA.model.framework.VLM4A.GroundingPIPIV3 import GroundingPIPIV3
    class Interface(nn.Module):
        def __init__(self):
            super().__init__()
            self.model=nn.Module()
            self.model.emb=nn.Embedding(16,16)
            self.model.get_input_embeddings=lambda:self.model.emb
            self.model.config=SimpleNamespace(hidden_size=16,text_config=SimpleNamespace(num_hidden_layers=2))
        def build_qwenvl_inputs(self,images,instructions):
            assert len(images)==1 and len(images[0])==3
            bins=instructions[0].split('[STATE] ')[1].split(' [ACTION]')[0].split()
            assert len(bins)==14 and all(0<=int(x)<=255 for x in bins)
            return {'input_ids':torch.arange(4)[None],'attention_mask':torch.ones(1,4,dtype=torch.bool)}
        def forward(self,input_ids,**kwargs):
            hidden=self.model.emb(input_ids)
            return SimpleNamespace(hidden_states=(hidden,hidden,hidden))
    monkeypatch.setattr(v3,'get_vlm_model',lambda config:Interface())
    cfg=OmegaConf.load(SCRIPT.with_name('groundingpipi_v3_clean.yaml'))
    cfg.framework.action_model.diffusion_model_cfg.action_dit_hidden_dim=8
    cfg.framework.action_model.diffusion_model_cfg.attention_head_dim=4
    model=GroundingPIPIV3(cfg).eval()
    assert model.action_model.state_encoder is None
    sample={'image':[Image.new('RGB',(224,224)) for _ in range(3)],'lang':'pick bottle',
            'state':np.linspace(-1,1,14,dtype=np.float32)[None],
            'action':np.zeros((16,14),np.float32)}
    with torch.no_grad():
        loss=model([sample])['action_loss']
        pred=model.predict_action([sample])['normalized_actions']
    assert torch.isfinite(loss).all()
    assert pred.shape==(1,16,14) and np.isfinite(pred).all()


def test_accumulation_callbacks_only_run_after_optimizer_step(monkeypatch):
    import torch
    from unittest.mock import Mock
    monkeypatch.setenv('STARVLA_DISABLE_DEEPSPEED','1')
    from starVLA.training.train_starvla import VLATrainer
    trainer=VLATrainer.__new__(VLATrainer)
    trainer.model=torch.nn.Linear(1,1)
    trainer.config=OmegaConf.create({'trainer':{'max_train_steps':2,'eval_interval':2,'save_interval':2}})
    trainer.completed_steps=0
    trainer.accelerator=SimpleNamespace(sync_gradients=False,is_local_main_process=False)
    trainer._log_training_config=Mock()
    trainer._create_data_iterators=Mock()
    trainer._get_next_batch=Mock(return_value=None)
    calls=[]
    def step(batch):
        calls.append(None)
        trainer.accelerator.sync_gradients=len(calls)%2==0
        return {'action_dit_loss':1.}
    trainer._train_step=step
    trainer.eval_action_model=Mock(side_effect=lambda metrics:metrics)
    trainer._log_metrics=Mock()
    trainer._save_checkpoint=Mock()
    trainer._finalize_training=Mock()
    trainer.train()
    assert len(calls)==4 and trainer.completed_steps==2
    assert trainer._log_metrics.call_count==2
    assert trainer.eval_action_model.call_count==1
    assert trainer._save_checkpoint.call_count==1


def test_accelerate_and_deepspeed_accumulation_agree(monkeypatch):
    monkeypatch.setenv('STARVLA_DISABLE_DEEPSPEED','1')
    from accelerate.utils import DistributedType
    from starVLA.training.train_starvla import configure_gradient_accumulation
    cfg=OmegaConf.create({'trainer':{'gradient_accumulation_steps':4}})
    fake=SimpleNamespace(distributed_type=DistributedType.DEEPSPEED,
                         gradient_accumulation_steps=1,
                         gradient_state=SimpleNamespace(plugin_kwargs={}))
    configure_gradient_accumulation(cfg,fake)
    assert fake.gradient_accumulation_steps==4
    assert fake.gradient_state.plugin_kwargs['sync_each_batch'] is True
    assert fake.gradient_state.plugin_kwargs['sync_with_dataloader'] is False
    assert 8*4*fake.gradient_accumulation_steps==128
    cfg.trainer.gradient_accumulation_steps=0
    with pytest.raises(ValueError,match='positive'):
        configure_gradient_accumulation(cfg,fake)


def test_pi05_launch_and_normalized_state_contract(data, monkeypatch):
    import torch
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
    data.framework = 'pi05'
    data.num_warmup_steps = 1
    data.health_check_interval = 1
    data.save_interval = 3
    cfg = runner.config(data)
    assert cfg.framework.name == 'GroundingPIPI05'
    assert cfg.framework.action_model.state_dim == 14
    assert cfg.framework.action_model.num_inference_steps == 10
    assert DATASET_NAMED_MIXTURES[cfg.datasets.vla_data.data_mix] == [('clean', 1.0, 'groundingpi_pi05_robotwin')]
    transform = ROBOT_TYPE_CONFIG_MAP['groundingpi_pi05_robotwin'].transform().transforms[-1]
    value = transform({'state.left_joints': torch.tensor([-2.,0.,2.])})
    torch.testing.assert_close(value['state.left_joints'], torch.tensor([-1.,0.,1.]))
    manifest = runner.prepare(data)
    runner.write_json(runner.validation_path(data), {'passed':True, 'manifest':manifest,
        'model_contract':{'image_size':[224,224]}})
    runner.write_json(data.data_root/'dataset_statistics.json', {})
    data.model.mkdir(); (data.model/'config.json').write_text('{}')
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('dry run launched'))
    runner.train(data)
    command = (data.output/'launch_command.sh').read_text()
    assert '--framework pi05' in command and '--num-warmup-steps 1' in command
    assert OmegaConf.load(data.output/'launch.yaml').framework.name == 'GroundingPIPI05'


def test_pi05_oom_detection_only_retries_cuda_oom():
    path = SCRIPT.with_name('run_groundingpi_pi05_pipeline.py')
    spec = importlib.util.spec_from_file_location('robotwin_pi05_pipeline', path)
    pipeline = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipeline)
    assert pipeline.is_oom('torch.OutOfMemoryError: CUDA out of memory')
    assert not pipeline.is_oom('ValueError: invalid state')
    assert not pipeline.is_oom('DataLoader worker killed by signal')


def test_ki_config_is_opt_in_and_uses_pi05(data):
    data.framework = 'pi05'
    baseline = runner.config(data)
    assert not baseline.framework.get('knowledge_insulation', False)
    data.knowledge_insulation = True
    cfg = runner.config(data)
    assert cfg.framework.name == 'GroundingPIPI05'
    assert cfg.framework.knowledge_insulation
    assert Path(cfg.framework.action_model.fast_tokenizer).is_dir()
    assert cfg.datasets.vla_data.action_type == 'abs_qpos'
    assert cfg.trainer.max_train_steps == 10000
    assert cfg.framework.action_model.action_horizon == 16
    data.framework = 'pi_v3'
    with pytest.raises(ValueError, match='requires pi05'):
        runner.config(data)
