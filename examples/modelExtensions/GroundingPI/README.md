# GroundingPI for StarVLA

The native GroundingPI vision-language model can drive three action frameworks:

| `framework.name` | Action prediction |
| --- | --- |
| `GroundingPIPI` | Layer-wise flow matching, reusing QwenPI |
| `GroundingPIOFT` | Parallel continuous regression, reusing QwenOFT |
| `GroundingPIFast` | Autoregressive FAST action tokens, reusing QwenFast |

Run commands from the starVLA repository root. Environment setup requires `uv` on PATH, or `UV_BIN` pointing to an existing uv binary. This integration loads model weights locally; it does not use the GroundingPI HTTP grounding service.

## Environment and weights

`setup.sh` creates a Python 3.12 environment and installs the SHA256-verified Transformers fork bundled with the GroundingPI source repository. The default environment is `/share/project/robot-rfm/robot-exp/usr/cjy/envs/starvla`. Set `ENV_DIR`, `GROUNDINGPI_SOURCE`, and `UV_BIN` to override paths. An existing environment is not overwritten.

```bash
bash examples/modelExtensions/GroundingPI/setup.sh
source /share/project/robot-rfm/robot-exp/usr/cjy/envs/starvla/bin/activate
HF_HUB_DISABLE_XET=1 python examples/modelExtensions/GroundingPI/download_weights.py
```

Configure your network proxy before installation/download. Package versions are recorded in the environment's `installed.freeze.txt`; model revisions are recorded in each model directory's `download_revision.json`. Use this dedicated environment instead of installing the root `requirements.txt`, whose Transformers pin targets older backbones.

## Configuration

Each framework has its own module under `starVLA/model/framework/VLM4A/`: `GroundingPIPI.py`, `GroundingPIOFT.py`, and `GroundingPIFast.py`. Shared defaults live in `_groundingpi_config.py`; the backbone adapter lives in `starVLA/model/modules/vlm/GroundingPI.py`.

Three example YAMLs are in `train_files/`. Select the desired framework, set `qwenvl.base_vlm` to the complete local GroundingPI model package, and configure dataset paths, action dimensions and horizon for your robot. `qwenvl.backbone: groundingpi` supports renamed weight directories. The examples use the existing `libero_all` dataset mixture and expect separately prepared LIBERO LeRobot data.

The adapter uses native chat templates, non-thinking prompts, image-grid expansion and left padding. Multiple cameras remain separate images in their original order. PI reads all language-model layers. OFT creates a single `<robot_action_query>` token. FAST adds 2048 ordered `<robot_action_i>` tokens and checks that this matches the action tokenizer vocabulary; it preserves GroundingPI's spatial-token IDs. Training and checkpoint reconstruction add the same tokens deterministically to the same pinned base package.

Inputs follow existing StarVLA conventions: `image` is a list of PIL images, `lang` is the instruction, `action` is `[T, action_dim]`, and optional `state` is `[1, state_dim]`. Training returns `action_loss`; inference returns `normalized_actions` shaped `[B, action_horizon, action_dim]`. FAST raises an error for empty action generation rather than returning a fabricated trajectory. New FAST action embeddings require training.

## Verification

```bash
python -m pytest tests/test_groundingpi.py -q
python -m examples.modelExtensions.GroundingPI.smoke --framework GroundingPIOFT
python -m examples.modelExtensions.GroundingPI.smoke --framework GroundingPIPI
python -m examples.modelExtensions.GroundingPI.smoke --framework GroundingPIFast
```

Run the GPU checks sequentially. Each uses official weights, a mixed single/dual-camera batch, backward gradient checks, action inference and a strict checkpoint reload. FAST additionally fits a synthetic zero-action trajectory with its output head and checks tokenizer round-tripping. This tests execution and token plumbing, not robot task accuracy. Logs/results and full smoke checkpoints go to `playground/groundingpi-smoke` and can consume tens of GB.

The example configurations target the existing StarVLA trainer. Prepared datasets are required for full training; smoke tests do not need a simulator or dataset. The native trainer can run without DeepSpeed using `STARVLA_DISABLE_DEEPSPEED=1`. PI uses the full 36-layer, 2560-wide action expert, so full optimizer training may require sharding across GPUs even when smoke checks fit on one H100.

## Validated run

Validated on NVIDIA H100 80GB with Python 3.12.15, PyTorch 2.7.1+cu126 and the bundled Transformers 5.7.0 fork (source revision `90dd8674248257e1115e531f2250fd85a32c3863`). GroundingPI model revision: `08d397c8769f7f10b96ff6d56ee6f56acb04181f`.

| Framework | Forward/backward | Batch inference | Strict reload and identical seeded output | Peak allocated GPU memory |
| --- | --- | --- | --- | --- |
| GroundingPIOFT | Passed | `[2, 16, 7]` | Passed | 19.93 GiB |
| GroundingPIPI | Passed | `[2, 16, 7]` | Passed | 44.46 GiB |
| GroundingPIFast | Passed | `[2, 16, 7]` | Passed | 20.23 GiB |

FAST has a non-parametric action tokenizer; gradients flow through the VLM, including its output head. The smoke fit learned the two-token synthetic trajectory without constrained decoding. All tests used SDPA and `TORCHDYNAMO_DISABLE=1`; the shell smoke runner sets this default to avoid compilation overhead. Direct smoke commands can use the same environment variable.

The training environment installs PyTorch3D's Python transforms from pinned upstream source with native extensions disabled. The examples use `torchvision_av` for video, avoiding incompatible prebuilt Decord/PyTorch3D wheels on Python 3.12. Full dataset training needs prepared data; this validation did not evaluate task success or convergence on a robot dataset.
