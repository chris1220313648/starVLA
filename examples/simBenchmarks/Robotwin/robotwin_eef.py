"""Shared EEF20 geometry, image packing and normalization for train and serve."""
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

CAMERAS = ('head_camera', 'left_camera', 'right_camera')
ROTATION_DIMS = [*range(3, 9), *range(13, 19)]
EPS = 1e-6


def eef20(left_pose, right_pose, left_gripper, right_gripper, *, quaternion_order='xyzw'):
    if quaternion_order not in ('xyzw', 'wxyz'):
        raise ValueError('Unsupported quaternion_order')
    arms = []
    for pose, grip in ((left_pose, left_gripper), (right_pose, right_gripper)):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape[-1] != 7 or not np.isfinite(pose).all():
            raise ValueError('Expected xyz + finite quaternion')
        quaternion = pose[..., 3:]
        if quaternion_order == 'wxyz':
            quaternion = quaternion[..., [1, 2, 3, 0]]
        matrix = Rotation.from_quat(quaternion.reshape(-1, 4)).as_matrix().reshape(*pose.shape[:-1], 3, 3)
        rotation = np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)
        grip = np.asarray(grip, dtype=np.float64).reshape(*pose.shape[:-1], 1)
        arms.append(np.concatenate((pose[..., :3], rotation, grip), axis=-1))
    result = np.concatenate(arms, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite EEF state')
    return result


def to_ee16(action, *, quaternion_order='xyzw'):
    if quaternion_order not in ('xyzw', 'wxyz'):
        raise ValueError('Unsupported quaternion_order')
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (20,) or not np.isfinite(action).all():
        raise ValueError('Expected finite EEF action (20,)')
    arms = []
    for arm in (action[:10], action[10:]):
        a, b = arm[3:6], arm[6:9]
        # Predictions need not form a perfect basis; supply a stable orthogonal fallback.
        a = a / np.linalg.norm(a) if np.linalg.norm(a) > EPS else np.array([1., 0., 0.])
        b = b - np.dot(a, b) * a
        if np.linalg.norm(b) <= EPS:
            basis = np.eye(3)[np.argmin(np.abs(a))]
            b = basis - np.dot(a, basis) * a
        b /= np.linalg.norm(b)
        quaternion = Rotation.from_matrix(np.stack((a, b, np.cross(a, b)), axis=-1)).as_quat()
        if quaternion_order == 'wxyz':
            quaternion = quaternion[[3, 0, 1, 2]]
        arms.append(np.concatenate((arm[:3], quaternion, arm[9:10])))
    return np.concatenate(arms).astype(np.float32)


def compose_image(images):
    if len(images) != 3:
        raise ValueError('Expected head, left wrist, right wrist images')
    canvas = Image.new('RGB', (320, 384))
    for image, size, position in zip(images, ((320, 256), (160, 128), (160, 128)), ((0, 0), (0, 256), (160, 256))):
        image = image if isinstance(image, Image.Image) else Image.fromarray(image)
        canvas.paste(image.convert('RGB').resize(size, Image.Resampling.BILINEAR), position)
    return canvas


def normalize(values, stats):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.asarray(stats['min'], dtype=np.float32), np.asarray(stats['max'], dtype=np.float32)
    if values.shape[-1] != 20 or not np.isfinite(values).all():
        raise ValueError('Expected finite 20-D values')
    return np.clip(2 * (values - lo) / np.maximum(hi - lo, EPS) - 1, -1, 1).astype(np.float32)


def unnormalize(values, stats):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.asarray(stats['min'], dtype=np.float32), np.asarray(stats['max'], dtype=np.float32)
    return ((values + 1) * .5 * np.maximum(hi - lo, EPS) + lo).astype(np.float32)


class EEFNormProcessor:
    """Same min-max function and stats as the HDF5 training reader."""
    def __init__(self, ckpt_path, unnorm_key=None):
        from starVLA.model.framework.share_tools import read_mode_config
        _, stats = read_mode_config(str(ckpt_path))
        self.available_unnorm_keys = list(stats)
        self.unnorm_key = unnorm_key or (next(iter(stats)) if len(stats) == 1 else None)
        if self.unnorm_key not in stats:
            raise ValueError('Select a valid EEF normalization key')
        self.stats = stats[self.unnorm_key]
        self.action_keys = ['action.eef']
        self.state_keys = ['state.eef']

    def apply_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape not in ((20,), (1, 20)):
            raise ValueError('Expected EEF state (20,) or (1,20)')
        return normalize(state.reshape(1, 20), self.stats['state'])

    def unapply_actions(self, actions):
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[-1] != 20 or not np.isfinite(actions).all():
            raise ValueError('Expected finite EEF actions (T,20)')
        return unnormalize(actions, self.stats['action'])
