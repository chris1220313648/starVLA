"""Read original RoboTwin HDF5 endpose labels without materializing video copies."""
import bisect
import fcntl
import json
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from examples.simBenchmarks.Robotwin.robotwin_eef import CAMERAS, ROTATION_DIMS, compose_image, eef20, normalize


def read_eef(handle):
    return eef20(*(handle['endpose/' + key][:] for key in
                   ('left_endpose', 'right_endpose', 'left_gripper', 'right_gripper')))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(raw_root, cache_root):
    raw_root, cache_root = Path(raw_root).resolve(), Path(cache_root).resolve()
    if raw_root == cache_root or raw_root in cache_root.parents or cache_root in raw_root.parents:
        raise ValueError('Raw data and EEF cache must be separate')
    files = sorted(raw_root.glob('*/aloha-agilex_clean_50/data/episode*.hdf5'))
    if not files:
        raise ValueError('No RoboTwin Clean HDF5 episodes found')
    sources = []
    for path in files:
        instruction = path.parent.parent / 'instructions' / (path.stem + '.json')
        info, text_info = path.stat(), instruction.stat()
        sources.append(dict(path=str(path), size=info.st_size, mtime_ns=info.st_mtime_ns,
                            instruction=str(instruction), instruction_size=text_info.st_size,
                            instruction_mtime_ns=text_info.st_mtime_ns))
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root / 'prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = cache_root / 'index.json'
        if index.exists():
            saved = json.loads(index.read_text())
            if saved['sources'] != sources:
                raise ValueError('EEF source changed; choose a new cache directory')
            if not (cache_root / 'dataset_statistics.json').is_file():
                raise ValueError('Missing EEF statistics; use a new cache directory')
            return saved
        rows, arrays = [], []
        for source in sources:
            path = Path(source['path'])
            with h5py.File(path, 'r') as handle:
                values = read_eef(handle)
                n = len(values)
                if n < 2:
                    raise ValueError(f'Need >=2 frames: {path}')
                for camera in CAMERAS:
                    if len(handle[f'observation/{camera}/rgb']) != n:
                        raise ValueError(f'Camera/action length mismatch: {path} {camera}')
            instructions = json.loads(Path(source['instruction']).read_text())['seen']
            if not instructions or not all(isinstance(text, str) and text.strip() for text in instructions):
                raise ValueError(f'No valid seen instructions: {path}')
            rows.append(dict(path=str(path), length=n, task=path.parents[2].name, instructions=instructions))
            arrays.append(values)
        values = np.concatenate(arrays)
        stat = {key: np.asarray(value).tolist() for key, value in dict(
            min=values.min(0), max=values.max(0), mean=values.mean(0), std=values.std(0)).items()}
        # Rotation basis components already lie in [-1,1]; preserve their geometry.
        for key, value in (('min', -1), ('max', 1), ('mean', 0), ('std', 1)):
            for dimension in ROTATION_DIMS:
                stat[key][dimension] = value
        result = dict(sources=sources, episodes=rows, total_frames=len(values),
                      total_samples=sum(row['length'] - 1 for row in rows),
                      action_dim=20, state_dim=20, action_horizon=32, image_hw=[384, 320])
        write_json(cache_root / 'dataset_statistics.json', {'robotwin_eef': dict(
            action=stat, state=stat, num_trajectories=len(rows), num_transitions=result['total_samples'])})
        write_json(index, result)
        return result


class RobotwinEEFDataset(Dataset):
    def __init__(self, data_cfg, training=True):
        self.data_cfg = data_cfg
        self.training = training
        self.color_jitter = float(getattr(data_cfg, 'color_jitter', 0.0))
        self.filter_static_segments = bool(getattr(data_cfg, 'filter_static_segments', False))
        self.static_segment_threshold = float(getattr(data_cfg, 'static_segment_threshold', 1e-5))
        self.max_static_retry = int(getattr(data_cfg, 'max_static_retry', 3))
        if not 0 <= self.color_jitter <= 1 or not self.static_segment_threshold > 0 or self.max_static_retry < 0:
            raise ValueError('Invalid color jitter or static resampling settings')
        self.cache_root = Path(data_cfg.data_root_dir)
        self.index = json.loads((self.cache_root / 'index.json').read_text())
        self.episodes = self.index['episodes']
        self.ends = np.cumsum([row['length'] - 1 for row in self.episodes]).tolist()
        self.stats = json.loads((self.cache_root / 'dataset_statistics.json').read_text())
        self._last_episode = None
        self._last_values = None

    def __len__(self):
        return self.ends[-1]

    def save_dataset_statistics(self, path):
        write_json(path, self.stats)

    def _window(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self.ends, index)
        episode = self.episodes[episode_index]
        frame = index - (self.ends[episode_index - 1] if episode_index else 0)
        if self._last_episode != episode_index:
            with h5py.File(episode['path'], 'r') as handle:
                self._last_values = read_eef(handle)
            self._last_episode = episode_index
        return episode, frame, self._last_values

    def _is_static(self, window):
        _, frame, values = window
        pair = normalize(values[frame:frame + 2], self.stats['robotwin_eef']['action'])
        return np.max(np.abs(pair[1] - pair[0])) < self.static_segment_threshold

    def __getitem__(self, index):
        window = self._window(index)
        if self.training and self.filter_static_segments and len(self) > 1 and self._is_static(window):
            for _ in range(self.max_static_retry):
                candidate = self._window(random.randrange(len(self)))
                if not self._is_static(candidate):
                    window = candidate
                    break
        episode, frame, values = window
        with h5py.File(episode['path'], 'r') as handle:
            images = []
            for camera in CAMERAS:
                encoded = handle[f'observation/{camera}/rgb'][frame]
                # RoboTwin passes RGB to OpenCV's BGR encoder; OpenCV decoding
                # restores the original RGB array. Do not apply BGR2RGB here.
                decoded = cv2.imdecode(np.frombuffer(bytes(encoded), np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    raise ValueError(f'Invalid JPEG: {episode["path"]} {camera} frame {frame}')
                images.append(Image.fromarray(decoded))
        positions = frame + 1 + np.arange(32)
        valid = positions < len(values)
        actions = values[np.minimum(positions, len(values) - 1)]
        stats = self.stats['robotwin_eef']
        image = compose_image(images)
        if self.training and self.color_jitter:
            for enhance in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                image = enhance(image).enhance(random.uniform(1 - self.color_jitter, 1 + self.color_jitter))
        return dict(image=[image],
                    state=normalize(values[frame:frame + 1], stats['state']),
                    action=normalize(actions, stats['action']), action_mask=valid,
                    lang=episode['instructions'][np.random.randint(len(episode['instructions']))])


def get_vla_dataset(data_cfg, training=True, **kwargs):
    return RobotwinEEFDataset(data_cfg, training=training)
