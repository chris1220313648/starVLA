"""LIBERO benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys), # igore state modality for now since some datasets don't have state and we want to be able to use them, can add back later if needed
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


class OpenPILibero4in1DataConfig(Libero4in1DataConfig):
    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys + self.action_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "q99",
                    "state.y": "q99",
                    "state.z": "q99",
                    "state.roll": "q99",
                    "state.pitch": "q99",
                    "state.yaw": "q99",
                    "state.pad": "q99",
                    "state.gripper": "q99",
                },
            ),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "q99",
                    "action.y": "q99",
                    "action.z": "q99",
                    "action.roll": "q99",
                    "action.pitch": "q99",
                    "action.yaw": "q99",
                    "action.gripper": "q99",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "openpi_libero_franka": OpenPILibero4in1DataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "openpi_libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "openpi_libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "openpi_libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "openpi_libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "openpi_libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}


class LiberoV3DataConfig(Libero4in1DataConfig):
    """GroundingPI contract: two cameras, raw 8-D state, 16 delta actions."""
    action_indices = list(range(16))


from starVLA.dataloader.gr00t_lerobot.transform.base import ModalityTransform


class ClipNormalizedState(ModalityTransform):
    def apply(self, data):
        for key in self.apply_to:
            if key in data:
                data[key] = data[key].clamp(-1, 1)
        return data


class GroundingPIV3GoalDataConfig(LiberoV3DataConfig):
    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys,
                                 normalization_modes={key: 'min_max' for key in self.state_keys}),
            ClipNormalizedState(apply_to=self.state_keys),
        ]
        return ComposedModalityTransform(transforms=transforms + super().transform().transforms)


ROBOT_TYPE_CONFIG_MAP['groundingpi_v3_goal_franka'] = GroundingPIV3GoalDataConfig()
DATASET_NAMED_MIXTURES['groundingpi_v3_libero_goal'] = [('libero_goal', 1.0, 'groundingpi_v3_goal_franka')]


ROBOT_TYPE_CONFIG_MAP["libero_v3_franka"] = LiberoV3DataConfig()
_LIBERO_V3_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
for _suite in _LIBERO_V3_SUITES:
    DATASET_NAMED_MIXTURES[f"{_suite}_v3"] = [(_suite, 1.0, "libero_v3_franka")]
DATASET_NAMED_MIXTURES["libero_v3_four"] = [
    (suite, 1.0, "libero_v3_franka") for suite in _LIBERO_V3_SUITES[:4]
]
DATASET_NAMED_MIXTURES["libero_v3_five"] = [
    (suite, 1.0, "libero_v3_franka") for suite in _LIBERO_V3_SUITES
]

# PI05 discretizes normalized state; keep the raw-state PI data contract intact.
ROBOT_TYPE_CONFIG_MAP['groundingpi_pi05_franka'] = GroundingPIV3GoalDataConfig()
for _suite in _LIBERO_V3_SUITES:
    DATASET_NAMED_MIXTURES[f'groundingpi_pi05_{_suite}'] = [(_suite, 1.0, 'groundingpi_pi05_franka')]
for _name, _suites in [('four', _LIBERO_V3_SUITES[:4]), ('five', _LIBERO_V3_SUITES)]:
    DATASET_NAMED_MIXTURES[f'groundingpi_pi05_{_name}'] = [(s, 1.0, 'groundingpi_pi05_franka') for s in _suites]
