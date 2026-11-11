"""Read EEF20 LeRobot v3 and attach optional source-aligned grounding labels."""
import bisect
from collections import OrderedDict
import fcntl
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from examples.simBenchmarks.Robotwin.robotwin_eef import EPS, ROTATION_DIMS, compose_image, normalize
from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_dataset import write_json

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
DEFAULT_DATASET = Path('/share/project/robot-rfm/robot-processed-data/lerobot_v3_data/robotwin2.0_EEF20_Lerobotv3')
DEFAULT_COT = Path('/share/project/robot-rfm/robot-processed-data/lerobot_v3_data/robotwin2.0_Lerobotv3_cot/clean/20261029_v1')


COMPOSITE_COT_PROMPT = (
    'The image is a three-view composite: the head view occupies the full width of the top two thirds; '
    'the left and right wrist views are at the bottom. Locate only the current or next directly '
    'manipulated objects or parts in the TOP HEAD VIEW for this task, including held objects. '
    'Return GroundingPI bbox entries only, without arm labels, using xyxy coordinates on the '
    '0-999 grid of the ENTIRE composite image, not the head view alone.'
)

SEPARATE_COT_PROMPT = (
    'Three images are provided in order: head camera, left wrist camera, right wrist camera. '
    'Locate only the current or next directly manipulated objects or parts in the FIRST (head camera) '
    'image for this task, including held objects. Return GroundingPI bbox entries only, without arm '
    'labels, using xyxy coordinates on the 0-999 grid of the FIRST image.'
)


def map_cot_to_composite(answer):
    """Map native head-image bbox tokens into the top 256 rows of a 384-row canvas."""
    entry = re.compile(r'<\|object_ref_start\|>[^<>]+<\|object_ref_end\|>'
                       r'<\|box_start\|>((?:<\d+>){4}(?:\s*,\s*(?:<\d+>){4})*)<\|box_end\|>')
    coordinates = re.compile(r'<(\d+)><(\d+)><(\d+)><(\d+)>')

    def convert(match):
        x1, y1, x2, y2 = map(int, match.groups())
        if not (0 <= x1 <= x2 <= 999 and 0 <= y1 <= y2 <= 999):
            raise ValueError('Invalid target-only CoT bbox coordinates')
        return f'<{x1}><{(2*y1+1)//3}><{x2}><{(2*y2+1)//3}>'

    cursor, pieces = 0, []
    for match in entry.finditer(answer):
        separator = answer[cursor:match.start()].strip()
        if separator and (cursor == 0 or separator != ','):
            raise ValueError('Invalid target-only CoT bbox syntax')
        start = match.group(0).split('<|box_start|>')[0]
        pieces.append(answer[cursor:match.start()] + start + '<|box_start|>' +
                      coordinates.sub(convert, match.group(1)) + '<|box_end|>')
        cursor = match.end()
    if not pieces or answer[cursor:].strip():
        raise ValueError('Invalid target-only CoT bbox syntax')
    return ''.join(pieces) + answer[cursor:]


def episode_metadata(root):
    rows = pd.concat([pd.read_parquet(p) for p in sorted((root / 'meta/episodes').rglob('*.parquet'))])
    if rows.episode_index.duplicated().any():
        raise ValueError(f'Duplicate episode IDs: {root}')
    return {int(row['episode_index']): row for row in rows.to_dict('records')}


def sample_count(source):
    """New RGB tables already omit the terminal observation; legacy tables retain it."""
    length = int(source['length'])
    if 'raw_length' in source:
        if int(source['raw_length']) != length + 1:
            raise ValueError('Invalid raw_length for next-frame EEF transitions')
        count = length
    else:
        count = length - 1
    if count < 1:
        raise ValueError('EEF episode has no action transitions')
    return count


class ShardCache:
    """Bound memory independently of the number of visited episodes."""
    def __init__(self, columns=None):
        self.columns = columns
        self.tables = OrderedDict()

    def read(self, path):
        if path not in self.tables:
            table = pd.read_parquet(path, columns=self.columns)
            self.tables[path] = (table, table.groupby('episode_index', sort=False).indices)
            if len(self.tables) > 2:
                self.tables.popitem(last=False)
        self.tables.move_to_end(path)
        return self.tables[path]

    def episode(self, path, episode_id):
        table, groups = self.read(path)
        return table.iloc[groups[episode_id]]


class CoTAnnotations:
    def __init__(self, root, sources, image_input_mode='composite'):
        if image_input_mode not in ('composite', 'separate'):
            raise ValueError(f'Unsupported image_input_mode: {image_input_mode}')
        self.image_input_mode = image_input_mode
        self.root = Path(root).resolve()
        self.dataset = self.root / 'datasets/groundingpi_target_only'
        self.info = json.loads((self.dataset / 'meta/info.json').read_text())
        contract = json.loads((self.dataset / 'meta/cot_contract.json').read_text())
        if contract['profile'] != 'groundingpi_target_only' or contract['bbox_camera'] != 'observation.images.cam_high':
            raise ValueError('Expected target-only CoT on the current head camera')
        mapping = pd.read_parquet(self.root / 'shared/annotations/episode_mapping.parquet')
        self.episodes = episode_metadata(self.dataset)
        lookup = {}
        for row in mapping.to_dict('records'):
            key = (row['source_task'], int(row['source_episode']))
            if key in lookup:
                raise ValueError(f'Duplicate CoT source: {key}')
            lookup[key] = row
        self.mapping = {}
        for source in sources:
            key = (source['source_task'], int(source['local_episode_index']))
            row = lookup.get(key)
            if row is None or int(row['frames']) != sample_count(source):
                raise ValueError(f'CoT source/length mismatch: {key}')
            episode = self.episodes[int(row['episode_index'])]
            if int(episode['length']) != int(row['frames']):
                raise ValueError(f'CoT metadata length mismatch: {key}')
            self.mapping[int(source['episode_index'])] = int(row['episode_index'])
        if len(set(self.mapping.values())) != len(sources):
            raise ValueError('CoT mapping must be one-to-one')
        self.annotation_root = self.dataset / 'annotations'
        if not self.annotation_root.is_dir():
            self.annotation_root = self.root / 'shared/annotations'
        self.answers = ShardCache(['episode_index', 'frame_index', 'index', 'annotation.cot', 'annotation.cot_available'])
        self.qc = ShardCache(['episode_index', 'frame_index', 'index', 'metadata_json', 'target_only_metadata_json'])

    def sample(self, episode_id, frame):
        cot_id = self.mapping[episode_id]
        meta = self.episodes[cot_id]
        relative = self.info['data_path'].format(chunk_index=int(meta['data/chunk_index']), file_index=int(meta['data/file_index']))
        row = self.answers.episode(self.dataset / relative, cot_id).iloc[frame]
        qc = self.qc.episode(self.annotation_root / relative, cot_id).iloc[frame]
        identity = (cot_id, frame, int(meta['dataset_from_index']) + frame)
        for value in (row, qc):
            if tuple(int(value[k]) for k in ('episode_index', 'frame_index', 'index')) != identity:
                raise ValueError('CoT frame identity mismatch')
        metadata = json.loads(qc['metadata_json'])
        answer = row['annotation.cot']
        available = bool(row['annotation.cot_available']) and isinstance(answer, str) and bool(answer.strip())
        coordinate_frame = 'composite_320x384_v1' if self.image_input_mode == 'composite' else 'head_camera_320x240_v1'
        if available and self.image_input_mode == 'composite':
            answer = map_cot_to_composite(answer)
        return dict(cot=answer if available else None,
                    cot_coordinate_frame=coordinate_frame, lang=metadata['cot_primary_instruction'],
                    cot_metadata=json.loads(qc['target_only_metadata_json']), cot_episode_index=cot_id)


    def validate(self):
        identities = ShardCache(['episode_index', 'frame_index', 'index'])
        eligible = 0
        for cot_id in sorted(self.mapping.values()):
            ep = self.episodes[cot_id]
            relative = self.info['data_path'].format(chunk_index=int(ep['data/chunk_index']), file_index=int(ep['data/file_index']))
            rows = self.answers.episode(self.dataset / relative, cot_id)
            qc = identities.episode(self.annotation_root / relative, cot_id)
            for table in (rows, qc):
                np.testing.assert_array_equal(table.frame_index, np.arange(int(ep['length'])))
                np.testing.assert_array_equal(table['index'], np.arange(int(ep['dataset_from_index']), int(ep['dataset_from_index']) + int(ep['length'])))
            valid = rows['annotation.cot_available'] & rows['annotation.cot'].fillna('').str.strip().ne('')
            for answer in rows.loc[valid, 'annotation.cot'].unique():
                map_cot_to_composite(answer)  # validate native bbox syntax in either input mode
            eligible += int(valid.sum())
        return eligible


def source_fingerprint(root, cot_root, image_input_mode='composite', instruction_paths=()):
    paths = [root / 'source_episodes.jsonl', *sorted((root / 'meta').rglob('*')),
             *sorted((root / 'data').rglob('*.parquet')), *sorted((root / 'videos').rglob('*.mp4'))]
    if cot_root is not None:
        paths += sorted(Path(cot_root).rglob('*.parquet'))
        paths += sorted((Path(cot_root) / 'datasets/groundingpi_target_only/meta').glob('*.json'))
        paths += [Path(cot_root) / 'datasets/groundingpi_target_only/meta/model_visible_setup.txt']
    digest = hashlib.sha256()
    if instruction_paths:
        digest.update(b'instruction_source=seen100_v1')
        for path in instruction_paths:
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
    if cot_root is not None:
        digest.update(('composite_320x384_v1' if image_input_mode == 'composite' else 'head_camera_320x240_v1').encode())
    for path in paths:
        if path.is_file():
            stat = path.stat()
            digest.update(f'{path}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()


def prepare(dataset_root, cache_root, cot_root=None, image_input_mode='composite', instruction_source='default'):
    if instruction_source not in ('default', 'seen'):
        raise ValueError(f'Unsupported instruction_source: {instruction_source}')
    if image_input_mode not in ('composite', 'separate'):
        raise ValueError(f'Unsupported image_input_mode: {image_input_mode}')
    root, cache = (Path(dataset_root) / 'clean').resolve(), Path(cache_root).resolve()
    protected = [root.parent] + ([Path(cot_root).resolve()] if cot_root is not None else [])
    if any(cache == p or p in cache.parents or cache in p.parents for p in protected):
        raise ValueError('Source datasets and writable cache must be separate')
    info = json.loads((root / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0' or info['fps'] != 50:
        raise ValueError('Expected EEF LeRobot v3 at 50 FPS')
    expected_names = [f'{arm}_{name}' for arm in ('left', 'right') for name in
                      ('x', 'y', 'z', 'r00', 'r10', 'r20', 'r01', 'r11', 'r21', 'gripper')]
    for key in ('observation.state', 'action'):
        feature = info['features'][key]
        if feature['shape'] != [20] or feature.get('names') not in (expected_names, [expected_names]):
            raise ValueError(f'Expected left/right EEF20 column order: {key}')
    sources = [json.loads(line) for line in (root / 'source_episodes.jsonl').read_text().splitlines()]
    if len({int(r['episode_index']) for r in sources}) != len(sources):
        raise ValueError('Duplicate EEF source episode IDs')
    instruction_paths, seen_instructions = [], {}
    if instruction_source == 'seen':
        for source in sources:
            path = Path(source['instruction'])
            texts = json.loads(path.read_text()).get('seen')
            if (not isinstance(texts, list) or len(texts) != 100 or
                    any(not isinstance(text, str) or not text.strip() for text in texts)):
                raise ValueError(f'Expected 100 valid seen instructions: {path}')
            instruction_paths.append(path)
            seen_instructions[int(source['episode_index'])] = texts
    cot = CoTAnnotations(cot_root, sources, image_input_mode) if cot_root is not None else None
    fingerprint = source_fingerprint(root, cot_root, image_input_mode, instruction_paths)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = cache / 'index.json'
        if index_path.exists():
            saved = json.loads(index_path.read_text())
            if saved.get('source_fingerprint') != fingerprint:
                raise ValueError('EEF/CoT source changed; choose a new cache directory')
            if not (cache / 'dataset_statistics.json').is_file():
                raise ValueError('Missing EEF statistics; choose a new cache directory')
            return saved
        cot_eligible = cot.validate() if cot is not None else 0
        metadata = episode_metadata(root)
        if len(sources) != info['total_episodes'] or set(metadata) != {int(s['episode_index']) for s in sources}:
            raise ValueError('EEF source/metadata episodes differ')
        tables = ShardCache()
        rows, moments = [], {key: [] for key in ('state', 'action')}
        for source in sorted(sources, key=lambda s: s['episode_index']):
            eid, n = int(source['episode_index']), int(source['length'])
            samples = sample_count(source)
            ep = metadata[eid]
            path = root / info['data_path'].format(chunk_index=int(ep['data/chunk_index']), file_index=int(ep['data/file_index']))
            frame = tables.episode(path, eid)
            if len(frame) != n or int(ep['length']) != n:
                raise ValueError(f'EEF episode length mismatch: {eid}')
            np.testing.assert_array_equal(frame.frame_index, np.arange(n))
            np.testing.assert_array_equal(frame['index'], np.arange(int(ep['dataset_from_index']), int(ep['dataset_to_index'])))
            np.testing.assert_allclose(frame.timestamp, np.arange(n) / 50, atol=1e-5)
            values = {}
            for key, column in (('state', 'observation.state'), ('action', 'action')):
                value = np.stack(frame[column]).astype(np.float64)
                if value.shape != (n, 20) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid {column}: episode {eid}')
                values[key] = value
                moments[key].append((n, value.min(0), value.max(0), value.sum(0), np.square(value).sum(0)))
            np.testing.assert_allclose(values['action'][:-1], values['state'][1:], atol=1e-6)
            if samples < n:
                np.testing.assert_allclose(values['action'][-1], values['state'][-1], atol=1e-6)
            videos = {}
            for cam in CAMERAS:
                key = 'observation.images.' + cam
                prefix = 'videos/' + key
                video = root / info['video_path'].format(video_key=key, chunk_index=int(ep[prefix+'/chunk_index']), file_index=int(ep[prefix+'/file_index']))
                if not video.is_file():
                    raise FileNotFoundError(video)
                videos[cam] = dict(path=str(video), start=float(ep[prefix+'/from_timestamp']))
            rows.append(dict(episode_index=eid, length=n, sample_length=samples, task=source['source_task'],
                             path=str(path), videos=videos, instruction=source['task']))
            if instruction_source == 'seen':
                rows[-1]['seen_instructions'] = seen_instructions[eid]
        stats = {}
        for key, items in moments.items():
            count = sum(v[0] for v in items)
            mean = sum(v[3] for v in items) / count
            stat = dict(min=np.min([v[1] for v in items], axis=0), max=np.max([v[2] for v in items], axis=0),
                        mean=mean, std=np.sqrt(np.maximum(sum(v[4] for v in items) / count - mean**2, 0)))
            for name, value in (('min', -1), ('max', 1), ('mean', 0), ('std', 1)):
                stat[name][ROTATION_DIMS] = value
            stats[key] = {name: value.tolist() for name, value in stat.items()}
        total_frames = sum(row['length'] for row in rows)
        if total_frames != info['total_frames']:
            raise ValueError('EEF total frame count mismatch')
        image_hw = [384, 320] if image_input_mode == 'composite' else [240, 320]
        result = dict(source=str(root), source_fingerprint=fingerprint, episodes=rows, total_frames=total_frames,
                      total_samples=sum(row['sample_length'] for row in rows), action_dim=20, state_dim=20, action_horizon=32,
                      image_hw=image_hw, quaternion_order='wxyz', cot_root=str(Path(cot_root).resolve()) if cot else None,
                      cot_episode_mapping=cot.mapping if cot else {}, cot_eligible_frames=cot_eligible,
                      cot_coordinate_frame=(('composite_320x384_v1' if image_input_mode == 'composite' else 'head_camera_320x240_v1') if cot else None),
                      image_input_mode=image_input_mode, instruction_source=instruction_source)
        write_json(cache / 'dataset_statistics.json', {'robotwin_eef': dict(**stats, num_trajectories=len(rows), num_transitions=result['total_samples'])})
        write_json(index_path, result)
        return result


def image_frame_indices(frame, history_frames=0, history_stride=40):
    if history_frames == 0:
        return [frame]
    return [0] + [frame - offset if frame >= offset else frame
                  for offset in range((history_frames - 1) * history_stride, -1, -history_stride)]


class RobotwinEEFLeRobotDataset(Dataset):
    def __init__(self, data_cfg, training=True):
        self.data_cfg, self.training = data_cfg, training
        self.index = json.loads((Path(data_cfg.data_root_dir) / 'index.json').read_text())
        self.episodes = self.index['episodes']
        self.instruction_source = getattr(data_cfg, 'instruction_source', 'default')
        if self.instruction_source != self.index.get('instruction_source', 'default'):
            raise ValueError('Dataset config and prepared instruction_source disagree')
        self.ends = np.cumsum([r.get('sample_length', r['length']-1) for r in self.episodes]).tolist()
        self.stats = json.loads((Path(data_cfg.data_root_dir) / 'dataset_statistics.json').read_text())
        self.tables = ShardCache(['episode_index', 'frame_index', 'observation.state', 'action'])
        self.color_jitter = float(getattr(data_cfg, 'color_jitter', 0))
        self.filter_static_segments = bool(getattr(data_cfg, 'filter_static_segments', False))
        self.threshold = float(getattr(data_cfg, 'static_segment_threshold', 1e-5))
        self.retries = int(getattr(data_cfg, 'max_static_retry', 3))
        if not 0 <= self.color_jitter <= 1 or self.threshold <= 0 or self.retries < 0:
            raise ValueError('Invalid augmentation/resampling settings')
        self.history_frames = int(getattr(data_cfg, 'history_frames', 0))
        self.history_stride = int(getattr(data_cfg, 'history_stride', 40))
        if self.history_frames not in (0, 4) or self.history_stride <= 0:
            raise ValueError('History requires four window frames and a positive stride')
        if self.history_frames and getattr(data_cfg, 'image_input_mode', 'composite') != 'composite':
            raise ValueError('History requires composite images')
        self.cot = None
        if self.index['cot_root']:
            source = Path(self.index['source']) / 'source_episodes.jsonl'
            self.cot = CoTAnnotations(self.index['cot_root'], [json.loads(s) for s in source.read_text().splitlines()],
                                      self.index.get('image_input_mode', 'composite'))
        self.image_input_mode = getattr(data_cfg, 'image_input_mode', self.index.get('image_input_mode', 'composite'))
        if self.image_input_mode != self.index.get('image_input_mode', 'composite'):
            raise ValueError('Dataset config and prepared image_input_mode disagree')

    def __len__(self):
        return self.ends[-1]

    def save_dataset_statistics(self, path):
        write_json(path, self.stats)

    def _window(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ei = bisect.bisect_right(self.ends, index)
        ep = self.episodes[ei]
        frame = index - (self.ends[ei-1] if ei else 0)
        return ep, frame, self.tables.episode(Path(ep['path']), ep['episode_index'])

    def _static(self, window):
        _, frame, rows = window
        stats = self.stats['robotwin_eef']['state']
        scale = np.maximum(np.asarray(stats['max'])-np.asarray(stats['min']), EPS)
        # The final transition target can exceed the observed-state range; do not clip it.
        delta = np.asarray(rows.action.iloc[frame])-np.asarray(rows['observation.state'].iloc[frame])
        return np.max(2*np.abs(delta)/scale) < self.threshold

    def __getitem__(self, index):
        from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps
        window = self._window(index)
        if self.training and self.filter_static_segments and self._static(window):
            for _ in range(self.retries):
                candidate = self._window(random.randrange(len(self)))
                if not self._static(candidate):
                    window = candidate
                    break
        ep, frame, rows = window
        indices = image_frame_indices(frame, self.history_frames, self.history_stride)
        unique = sorted(set(indices))
        decoded = {cam: get_frames_by_timestamps(ep['videos'][cam]['path'],
                   [ep['videos'][cam]['start'] + step / 50 for step in unique], video_backend='pyav')
                   for cam in CAMERAS}
        frames = {step: [Image.fromarray(decoded[cam][i]) for cam in CAMERAS]
                  for i, step in enumerate(unique)}
        if self.image_input_mode == 'separate':
            image_sample = [image.convert('RGB').resize((320, 240), Image.Resampling.BILINEAR)
                            for image in frames[frame]]
        else:
            composites = {step: compose_image(images) for step, images in frames.items()}
            image_sample = [composites[step] for step in indices]
        if self.training and self.color_jitter:
            for enhance in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                factor = random.uniform(1-self.color_jitter, 1+self.color_jitter)
                image_sample = [enhance(image).enhance(factor) for image in image_sample]
        positions = frame + np.arange(32)
        samples = ep.get('sample_length', ep['length']-1)
        valid = positions < samples
        actions = np.stack(rows.action.iloc[np.minimum(positions, samples-1)])
        stats = self.stats['robotwin_eef']
        result = dict(image=image_sample, state=normalize(np.asarray(rows['observation.state'].iloc[frame])[None], stats['state']),
                      action=normalize(actions, stats['action']), action_mask=valid, lang=ep['instruction'],
                      episode_index=ep['episode_index'], frame_index=frame)
        if self.cot is not None:
            result.update(self.cot.sample(ep['episode_index'], frame))
        if self.instruction_source == 'seen':
            texts = ep['seen_instructions']
            result['lang'] = random.choice(texts) if self.training else texts[0]
        return result


def get_vla_dataset(data_cfg, training=True, **kwargs):
    return RobotwinEEFLeRobotDataset(data_cfg, training=training)
