"""Observation acceptance checks without a simulator or model."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

EVAL = Path(__file__).resolve().parents[1] / "examples/simBenchmarks/Robotwin/eval_files"
sys.path.insert(0, str(EVAL))
spec = importlib.util.spec_from_file_location("cached_scenes", EVAL / "check_cached_scenes.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ObservationChecks(unittest.TestCase):
    def test_images_and_state(self):
        observation = {
            "observation": {name: {"rgb": np.arange(36, dtype=np.uint8).reshape(3, 4, 3)}
                            for name in ("head_camera", "left_camera", "right_camera")},
            "joint_action": {"vector": np.zeros(14)},
            "endpose": {"left_endpose": np.zeros(7)},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = checker.check_observation(observation, output, "before")
            self.assertEqual(len(list(output.glob("*.png"))), 3)
            self.assertEqual(len(result["joint_state"]), 14)
            observation["joint_action"]["vector"][0] = np.nan
            with self.assertRaisesRegex(ValueError, "joint state"):
                checker.check_observation(observation, output, "after")
            observation["joint_action"]["vector"][0] = 0
            observation["observation"]["head_camera"]["rgb"][:] = 0
            with self.assertRaisesRegex(ValueError, "Invalid RGB"):
                checker.check_observation(observation, output, "after")


if __name__ == "__main__":
    unittest.main()
