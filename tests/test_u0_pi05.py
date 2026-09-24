import unittest

import numpy as np

from starVLA.model.framework.VLM4A.U0PI05 import U0PI05
from starVLA.model.framework.base_framework import _auto_import_framework_modules
from starVLA.model.tools import FRAMEWORK_REGISTRY


class U0PI05ContractTest(unittest.TestCase):
    def test_registry_and_state_prompt(self):
        _auto_import_framework_modules()
        self.assertIn("U0PI05", FRAMEWORK_REGISTRY._registry)
        prompt = U0PI05._state_instruction("pick up the cube", np.array([[0.0, 1.0, -1.0]]))
        self.assertTrue(prompt.startswith("pick up the cube [STATE] "))
        self.assertTrue(prompt.endswith(" [ACTION]"))

    def test_state_prompt_rejects_nonfinite_values(self):
        with self.assertRaises(ValueError):
            U0PI05._state_instruction("pick", np.array([[np.nan]]))


if __name__ == "__main__":
    unittest.main()
