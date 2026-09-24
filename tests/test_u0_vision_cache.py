from unittest.mock import Mock

import numpy as np
import pytest

from starVLA.dataloader.u0_vision_cache import CachedLiberoDataset, validate_codes


def test_cached_sample_skips_video_and_preserves_action_path(tmp_path):
    tokens = np.arange(24, dtype=np.int32).reshape(3, 2, 2, 2)
    np.save(tmp_path / 'episode_000007.npy', tokens)
    dataset = object.__new__(CachedLiberoDataset)
    dataset.cache_dir = tmp_path
    dataset._cached_episode = None
    dataset._grid_shape = [2, 2]
    dataset._trajectory_lengths = np.array([3])
    dataset._modality_keys = {'video': ['video.primary_image', 'video.wrist_image']}
    dataset.get_trajectory_index = Mock(return_value=0)
    dataset.get_step_data = Mock(return_value={'action': 'original'})
    dataset.transforms = Mock(side_effect=lambda x: x)
    dataset._pack_sample = Mock(side_effect=lambda data, image_codes: {**data, 'image_codes': image_codes})
    sample = dataset.get_training_sample(7, 1)
    dataset.get_step_data.assert_called_once_with(7, 1, skip_video=True)
    assert sample == {'action': 'original', 'image_codes': tokens[1].tolist()}
    assert dataset._cached_episode == 7
    with pytest.raises(ValueError, match='Invalid cached'):
        validate_codes(tokens[:, :1], 3, 2)
    with pytest.raises(ValueError, match='Invalid cached'):
        validate_codes(tokens.astype(np.float32), 3, 2)


def test_cache_identity_does_not_depend_on_text_tokenizer(tmp_path):
    from omegaconf import OmegaConf
    from starVLA.dataloader.u0_vision_cache import cache_recipe
    (tmp_path / 'config.yaml').write_text('n_embed: 131072\n')
    (tmp_path / 'model.ckpt').write_bytes(b'test-weights')
    cfg = OmegaConf.create({'framework': {'u0': {'vision_tokenizer': str(tmp_path), 'image_size': 224,
                                               'base_vlm': '/unused/text-model-a'}},
                            'datasets': {'vla_data': {'video_backend': 'torchvision_av'}}})
    identity = cache_recipe(cfg)
    cfg.framework.u0.base_vlm = '/unused/text-model-b'
    cfg.framework.u0.tokenizer_path = '/also-not-needed'
    assert cache_recipe(cfg) == identity
    cfg.framework.u0.image_size = 256
    assert cache_recipe(cfg) != identity


def test_libero_all_cache_covers_every_dataset():
    from omegaconf import OmegaConf
    from starVLA.dataloader.u0_vision_cache import dataset_specs, make_dataset
    from unittest.mock import patch
    cfg = OmegaConf.load('examples/simBenchmarks/LIBERO/train_files/u0_fast_libero_all.yaml')
    specs = dataset_specs(cfg)
    names = [name for name, _, _ in specs]
    assert names == [f'libero_{suite}_no_noops_1.0.0_lerobot' for suite in ('object', 'goal', 'spatial', '10')]
    assert [weight for _, weight, _ in specs] == [1.0] * 4
    with patch('starVLA.dataloader.lerobot_datasets.make_LeRobotSingleDataset') as create:
        for name in names:
            make_dataset(cfg, name)
        assert [call.args[1] for call in create.call_args_list] == names
        assert all(not call.kwargs['data_cfg'].get('u0_vision_cache_dir') for call in create.call_args_list)
    with pytest.raises(ValueError, match='Select a dataset'):
        make_dataset(cfg)  # Must not silently cache only the first subset.
