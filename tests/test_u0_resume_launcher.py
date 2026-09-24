"""CPU-only launcher check; intercept Python before cache preparation or training."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


class ResumeLauncherTest(unittest.TestCase):
    def test_arguments_and_checkpoint_discovery(self):
        root = Path(__file__).resolve().parents[1]
        script = root / 'examples/simBenchmarks/LIBERO/train_files/run_u0_fast_libero_all_zero2_resume_20k.sh'
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source = temp / 'steps_7500_model.safetensors'
            source.touch()
            calls = temp / 'calls.jsonl'
            python = temp / 'python'
            python.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                              'with open(os.environ["CAPTURE"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
            python.chmod(0o755)
            env = {**os.environ, 'RESUME_CHECKPOINT': str(source), 'RUN_ROOT_DIR': tmp,
                   'RUN_ID': 'check', 'STARVLA_PYTHON': str(python), 'LOG_DIR': tmp,
                   'CAPTURE': str(calls)}
            for _ in range(2):
                subprocess.run(['bash', str(script)], env=env, check=True, capture_output=True)
            args = json.loads(calls.read_text().splitlines()[-1])
            for key, value in {'--num_processes': '8', '--trainer.max_train_steps': '20000',
                               '--trainer.save_interval': '5000', '--trainer.gradient_accumulation_steps': '4',
                               '--datasets.vla_data.per_device_batch_size': '8',
                               '--trainer.is_resume': 'true', '--run_root_dir': tmp}.items():
                self.assertEqual(args[args.index(key) + 1], value)
            folder = temp / 'check/checkpoints'
            self.assertEqual((folder / source.name).resolve(), source)
            # Exercise the trainer's actual lookup without importing GPU dependencies.
            tree = ast.parse((root / 'starVLA/training/trainer_utils/trainer_tools.py').read_text())
            method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_get_latest_checkpoint')
            namespace = {'os': os, 're': re}
            exec(compile(ast.Module(body=[method], type_ignores=[]), '<checkpoint lookup>', 'exec'), namespace)
            trainer = SimpleNamespace(accelerator=SimpleNamespace(print=lambda *a: None))
            lookup = namespace['_get_latest_checkpoint']
            self.assertEqual(lookup(trainer, str(folder))[1], 7500)
            (folder / 'steps_10000_model.safetensors').touch()
            self.assertEqual(lookup(trainer, str(folder))[1], 10000)


if __name__ == '__main__':
    unittest.main()
