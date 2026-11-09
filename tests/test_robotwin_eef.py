import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import cv2
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from examples.simBenchmarks.Robotwin.robotwin_eef import compose_image, eef20, normalize, unnormalize, to_ee16
from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_dataset import prepare, RobotwinEEFDataset


def test_geometry_layout_and_normalization():
    q = Rotation.from_euler('xyz', [.3, -.4, .9]).as_quat()
    pose = np.r_[1., 2., 3., q]
    value = eef20(pose, pose, .25, .8)
    recovered = to_ee16(value)
    np.testing.assert_allclose(recovered[:3], pose[:3])
    np.testing.assert_allclose(Rotation.from_quat(recovered[3:7]).as_matrix(), Rotation.from_quat(q).as_matrix(), atol=1e-6)
    assert value.shape == (20,) and recovered.shape == (16,)
    assert np.isfinite(to_ee16(np.zeros(20))).all()
    degenerate=np.zeros(20);degenerate[6]=degenerate[16]=1
    assert np.isfinite(to_ee16(degenerate)).all()
    images = [Image.new('RGB', (20, 30), c) for c in ('red', 'green', 'blue')]
    canvas = compose_image(images)
    assert canvas.size == (320, 384)
    assert canvas.getpixel((100, 100)) == (255, 0, 0)
    assert canvas.getpixel((100, 300)) == (0, 128, 0)
    assert canvas.getpixel((200, 300)) == (0, 0, 255)
    stats = {'min': [-2.]*20, 'max': [4.]*20}
    np.testing.assert_allclose(unnormalize(normalize(value, stats), stats), value, atol=1e-6)
    stats['max'][0] = stats['min'][0]
    assert np.isfinite(normalize(value, stats)).all()


@pytest.fixture
def data(tmp_path):
    raw = tmp_path / 'raw'
    for i in range(2):
        root = raw / 'task' / 'aloha-agilex_clean_50'
        (root / 'data').mkdir(parents=True, exist_ok=True)
        (root / 'instructions').mkdir(exist_ok=True)
        (root / f'instructions/episode{i}.json').write_text(json.dumps({'seen': ['pick bottle']}))
        pose = np.tile([0., 0., 0., 0., 0., 0., 1.], (4, 1))
        pose[:, 0] = np.arange(4)
        with h5py.File(root / f'data/episode{i}.hdf5', 'w') as f:
            for arm in ('left', 'right'):
                f[f'endpose/{arm}_endpose'] = pose
                f[f'endpose/{arm}_gripper'] = np.linspace(0, 1, 4)
            for cam in ('head_camera', 'left_camera', 'right_camera'):
                # Match RoboTwin's RGB-array-to-OpenCV encoding, including its swap.
                rgb = np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8)
                success, encoded = cv2.imencode('.jpg', rgb)
                assert success
                f[f'observation/{cam}/rgb'] = np.array([encoded.tobytes()]*4, dtype='S1000')
    cache = tmp_path / 'cache'
    index = prepare(raw, cache)
    return raw, cache, index


def test_dataset_future_alignment_and_padding(data):
    raw, cache, index = data
    assert len(index['episodes']) == 2 and index['total_frames'] == 8
    assert prepare(raw, cache) == index
    dataset = RobotwinEEFDataset(SimpleNamespace(data_root_dir=cache))
    assert len(dataset) == 6
    row = dataset[0]
    assert row['state'].shape == (1,20) and row['action'].shape == (32,20)
    actual = unnormalize(row['action'], dataset.stats['robotwin_eef']['action'])
    np.testing.assert_allclose(actual[:4,0], [1,2,3,3], atol=1e-6)
    assert row['action_mask'].sum() == 3
    assert dataset[2]['action_mask'].sum() == 1
    assert dataset[3]['action_mask'].sum() == 3
    assert row['image'][0].size == (320,384)
    for pixel in ((100, 100), (100, 300), (200, 300)):
        red, green, blue = row['image'][0].getpixel(pixel)
        assert red > 250 and green < 5 and blue < 5
    with pytest.raises(IndexError):dataset[len(dataset)]
    path = next(raw.glob('*/aloha-agilex_clean_50/instructions/*.json'))
    path.write_text(json.dumps({'seen':['changed']}))
    with pytest.raises(ValueError, match='source changed'):prepare(raw, cache)


def test_eef_launch_config():
    from omegaconf import OmegaConf
    path = Path('examples/simBenchmarks/Robotwin/train_files/groundingpipi05_eef_clean.yaml')
    cfg = OmegaConf.load(path)
    assert cfg.framework.action_model.action_dim == cfg.framework.action_model.state_dim == 20
    assert cfg.framework.action_model.action_horizon == 32
    assert cfg.framework.action_model.state_representation == 'continuous_expert_token'
    assert cfg.datasets.vla_data.image_size == [384,320]
    assert cfg.trainer.max_train_steps == 50000


@pytest.mark.parametrize('swap_red_blue', [False, True])
def test_eef_client_sends_one_canvas_and_executes_ee(monkeypatch, swap_red_blue):
    from examples.simBenchmarks.Robotwin.eval_files import model2robotwin_interface as client_module
    pose = np.array([.1,.2,.3,0,0,0,1])
    action = eef20(pose,pose,.25,.75)
    requests = []
    class Client:
        def __init__(self,*args):pass
        def get_server_metadata(self):return {'action_chunk_size':32}
        def predict_action(self,request):
            requests.append(request)
            return {'data':{'actions':np.tile(action,(1,32,1))}}
    monkeypatch.setattr(client_module,'WebsocketClientPolicy',Client)
    model = client_module.get_model(dict(policy_ckpt_path='', include_state=True,
        action_type='ee', image_size=[384,320], swap_red_blue=swap_red_blue))
    executed=[]
    class Environment:
        take_action_cnt=0
        def get_instruction(self):return 'pick bottle'
        def take_action(self,action,**kwargs):executed.append((action,kwargs))
    observation={'observation':{key:{'rgb':np.full((30,40,3),(255,0,0),dtype=np.uint8)} for key in ('head_camera','left_camera','right_camera')},
                 'endpose':dict(left_endpose=pose,right_endpose=pose,left_gripper=.25,right_gripper=.75)}
    client_module.eval(Environment(),model,observation)
    assert requests[0]['examples'][0]['image'][0].shape==(384,320,3)
    canvas = requests[0]['examples'][0]['image'][0]
    for y, x in ((100, 100), (300, 100), (300, 200)):
        np.testing.assert_array_equal(canvas[y, x], (0, 0, 255) if swap_red_blue else (255, 0, 0))
    np.testing.assert_array_equal(observation['observation']['head_camera']['rgb'][0, 0], (255, 0, 0))
    np.testing.assert_array_equal(requests[0]['examples'][0]['state'],action[None])
    assert executed[0][0].shape==(16,) and executed[0][1]=={'action_type':'ee'}


def test_saved_stats_are_used_by_serving(data,monkeypatch):
    from starVLA.model.framework import share_tools
    from examples.simBenchmarks.Robotwin.robotwin_eef import EEFNormProcessor
    _,cache,_=data
    stats=json.loads((cache/'dataset_statistics.json').read_text())
    monkeypatch.setattr(share_tools,'read_mode_config',lambda path:({},stats))
    processor=EEFNormProcessor('unused')
    dataset=RobotwinEEFDataset(SimpleNamespace(data_root_dir=cache))
    row=dataset[0]
    raw=unnormalize(row['state'],stats['robotwin_eef']['state'])
    np.testing.assert_allclose(processor.apply_state(raw),row['state'],atol=1e-6)
    np.testing.assert_allclose(processor.unapply_actions(row['action']),unnormalize(row['action'],stats['robotwin_eef']['action']))
    with pytest.raises(ValueError):processor.apply_state(np.zeros(14))


def test_color_jitter_order_and_eval_bypass(data, monkeypatch):
    from PIL import ImageEnhance
    from examples.simBenchmarks.Robotwin.train_files import robotwin_eef_dataset as module
    _, cache, _ = data
    cfg = SimpleNamespace(data_root_dir=cache, color_jitter=.2, filter_static_segments=True)
    base = RobotwinEEFDataset(cfg, training=False)[0]
    factors = iter([.8, 1.2, .9]); ranges = []
    def uniform(lo, hi):
        ranges.append((lo, hi))
        return next(factors)
    monkeypatch.setattr(module.random, 'uniform', uniform)
    monkeypatch.setattr(module.random, 'randrange', lambda _: pytest.fail('Nonstatic window must not resample'))
    actual = RobotwinEEFDataset(cfg)[0]
    expected = base['image'][0]
    for enhance, factor in zip((ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color), (.8, 1.2, .9)):
        expected = enhance(expected).enhance(factor)
    np.testing.assert_array_equal(actual['image'][0], expected)
    assert ranges == [(.8, 1.2)] * 3
    for key in ('state', 'action', 'action_mask'):
        np.testing.assert_array_equal(actual[key], base[key])
    monkeypatch.setattr(module.random, 'uniform', lambda *_: pytest.fail('Evaluation must not augment'))
    np.testing.assert_array_equal(RobotwinEEFDataset(cfg, training=False)[0]['image'][0], base['image'][0])


@pytest.mark.parametrize('moving_candidate', [True, False])
def test_static_resampling_and_three_attempt_fallback(data, monkeypatch, moving_candidate):
    from examples.simBenchmarks.Robotwin.train_files import robotwin_eef_dataset as module
    _, cache, _ = data
    cfg = SimpleNamespace(data_root_dir=cache, filter_static_segments=True, max_static_retry=3)
    dataset = RobotwinEEFDataset(cfg)
    original_window = dataset._window
    def window(index):
        episode, frame, values = original_window(index)
        values = values.copy()
        # Window zero is static; window two remains moving when requested.
        if index != 2 or not moving_candidate:
            values[frame + 1] = values[frame]
        return episode, frame, values
    monkeypatch.setattr(dataset, '_window', window)
    assert dataset._is_static(window(0))
    calls = []
    def candidate(size):
        assert size == len(dataset)
        calls.append(size)
        return 2
    monkeypatch.setattr(module.random, 'randrange', candidate)
    sample = dataset[0]
    expected_frame = 2 if moving_candidate else 0
    episode, frame, values = window(expected_frame)
    np.testing.assert_array_equal(sample['state'], normalize(values[frame:frame+1], dataset.stats['robotwin_eef']['state']))
    assert sample['action_mask'].sum() == (1 if moving_candidate else 3)
    assert len(calls) == (1 if moving_candidate else 3)
    dataset.training = False
    calls.clear()
    dataset[0]
    assert not calls


def test_static_threshold_is_in_normalized_space(data):
    _, cache, _ = data
    dataset = RobotwinEEFDataset(SimpleNamespace(data_root_dir=cache))
    episode, frame, values = dataset._window(0)
    values = values.copy(); values[1] = values[0]
    values[1, 0] += 1e-5  # min/max range is 3: normalized delta is below 1e-5.
    assert dataset._is_static((episode, frame, values))
    values[1, 0] += 1e-4
    assert not dataset._is_static((episode, frame, values))


def test_eef_augmentation_flags_saved_in_config(tmp_path):
    from examples.simBenchmarks.Robotwin.train_files.groundingpi_robotwin import config
    args = SimpleNamespace(framework='pi05_eef', model=tmp_path, data_root=tmp_path,
        raw_data=tmp_path, output=tmp_path/'run', batch_size=16, workers=2, steps=50000,
        color_jitter=.2, filter_static_segments=True)
    cfg = config(args)
    assert cfg.datasets.vla_data.color_jitter == .2
    assert cfg.datasets.vla_data.filter_static_segments
    assert cfg.datasets.vla_data.static_segment_threshold == 1e-5
    assert cfg.datasets.vla_data.max_static_retry == 3
