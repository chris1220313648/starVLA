"""Validate the complete EEF index and sample real image/action windows."""
import json
import numpy as np
from transformers import AutoProcessor

from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_dataset import RobotwinEEFDataset, write_json


def validate(args):
    from examples.simBenchmarks.Robotwin.train_files.groundingpi_robotwin import prepare, config, validation_path
    manifest = prepare(args)
    tasks = {episode['task'] for episode in manifest['episodes']}
    counts = {task: sum(row['task'] == task for row in manifest['episodes']) for task in tasks}
    if len(tasks) != 50 or set(counts.values()) != {50}:
        raise ValueError('Formal EEF training requires all 50 tasks, 50 episodes each')
    write_json(validation_path(args), dict(passed=False, manifest=manifest))
    dataset_class = RobotwinEEFDataset
    if getattr(args, 'dataset_root', None) is not None:
        from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_lerobot_dataset import RobotwinEEFLeRobotDataset
        dataset_class = RobotwinEEFLeRobotDataset
    dataset = dataset_class(config(args).datasets.vla_data, training=False)
    separate = getattr(args, 'image_input_mode', 'composite') == 'separate'
    history_cfg = config(args).datasets.vla_data
    history = int(history_cfg.get('history_frames', 0))
    history_contract = dict(frames=history, stride=int(history_cfg.get('history_stride', 40)), include_first=True, pad='current', cot_target='current') if history else None
    expected_images = 5 if history else 3 if separate else 1
    expected_size = (320, 240) if separate else (320, 384)
    expected_cot_frame = 'head_camera_320x240_v1' if separate else 'composite_320x384_v1'
    # Include episode tails to check padding/masking as well as starts.
    starts = [0, *dataset.ends[:-1]]
    chosen = np.linspace(0, len(starts) - 1, args.sample_episodes, dtype=int)
    for i in chosen:
        for index in (starts[i], dataset.ends[i] - 1):
            row = dataset[index]
            assert row['state'].shape == (1, 20) and row['action'].shape == (32, 20)
            assert row['action_mask'].shape == (32,) and row['action_mask'].any()
            assert len(row['image']) == expected_images and all(image.size == expected_size for image in row['image'])
            assert np.isfinite(row['state']).all() and np.isfinite(row['action']).all()
            if getattr(args, 'instruction_source', 'default') == 'seen':
                assert row['lang'] == dataset.episodes[i]['seen_instructions'][0]
            if row.get('cot') is not None:
                assert 'cot_image' not in row and row['cot_coordinate_frame'] == expected_cot_frame
                assert row['cot_metadata']['answer_available']
    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    pixels = processor.image_processor(images=dataset[0]['image'], return_tensors='pt')
    tokens = [int(grid.prod()) // processor.image_processor.merge_size**2 for grid in pixels['image_grid_thw']]
    if len(tokens) != expected_images or not tokens or any(token <= 0 for token in tokens) or len(set(tokens)) != 1:
        raise ValueError(f'Expected {expected_images} equally sized image token grids, got {tokens}')
    write_json(validation_path(args), dict(passed=True, manifest=manifest, tasks=counts,
        sampled_episodes=len(chosen), visual_tokens=tokens,
        model_contract=dict(history=history_contract, image_size=[240, 320] if separate else [384, 320], image_count=expected_images,
                            image_layout='head_left_wrist_right_separate' if separate else 'head_top_wrists_bottom',
                            cot_coordinate_frame=expected_cot_frame if args.with_cot else None,
                            state_shape=[1, 20], action_shape=[32, 20])))
    print(json.dumps(dict(passed=True, episodes=len(starts), samples=len(dataset), visual_tokens=tokens)), flush=True)
