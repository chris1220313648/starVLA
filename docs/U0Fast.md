# U0Fast

U0Fast fine-tunes the entire Xiaomi-Robotics-U0-4B UNIS backbone to predict FAST
action tokens. The native IBQ visual tokenizer stays frozen in FP32/eval mode.
The language backbone uses BF16, including its existing input embedding and
output head. This integration does not use LoRA.

The LIBERO Goal preset consumes the current primary and wrist views at 224×224
and predicts eight 7D actions. FAST indices 0–2047 reuse text IDs
149595–151642; image/prompt/padding tokens are ignored in the causal loss.
The full vocabulary is retained in the loss denominator, but vocabulary logits
are computed in chunks only at supervised positions. Inputs that collide with
reserved action IDs are rejected.

Generation additionally enforces FAST's UTF-8 coefficient count: only streams
encoding exactly `action_horizon * action_dim` coefficients may terminate.
Split-byte BPE tokens remain supported. This constrains serialization, not
coefficient values; malformed or unterminated streams raise instead of silently
returning the upstream decoder's zero-action fallback.

## Local setup

The training launcher defaults to the independent conda environment `starvla`
at `/opt/conda/envs/starvla/bin/python` (`conda activate starvla`). The simulation
environment is `/root/nas/envs/starvla-libero/bin/python`. Evaluation wrappers
still default to the original `/root/nas/envs/starvla-u0/bin/python` venv;
set `STARVLA_PYTHON=/opt/conda/envs/starvla/bin/python` to use conda there too.
Set `LIBERO_PYTHON` to override the simulator interpreter.

The verified training stack uses Python 3.11, PyTorch 2.8.0+cu128,
torchvision 0.23.0+cu128, Transformers 4.57.0, Accelerate 1.8.1,
DeepSpeed 0.16.9, FlashAttention 2.8.3, NumPy 1.26.4 and PyArrow 19.0.1.
The smoke run directory records full `environment_training.txt` and
`environment_libero.txt` package snapshots.

Model weights default to
`/root/nas/code/Xiaomi-Robotics-U0/training/models/Xiaomi-Robotics-U0-4B`.
The model Python source is vendored inside starVLA; only checkpoint assets are
read from that directory. `framework.u0.base_vlm`, optional
`framework.u0.tokenizer_path`, `framework.u0.vision_tokenizer` and
`framework.action_model.fast_tokenizer_name` configure these assets.

Supported attention backends are `flash_attention_2` (default) and `eager`.
Upstream U0 SDPA omits Q/K normalization and is explicitly rejected. A local
compatibility patch treats negative DynamicCache capacity as unbounded.

## Training

Run commands from the starVLA root. Select available GPUs explicitly.

```bash
# Two-GPU, ten-optimizer-step integration smoke.
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 MAX_TRAIN_STEPS=10 SAVE_INTERVAL=10 \
RUN_ID=u0_fast_libero_goal_smoke \
bash examples/simBenchmarks/LIBERO/train_files/run_u0_fast_libero.sh \
  --trainer.num_warmup_steps 1

# Formal preset: 10,000 steps, per-GPU batch 1, accumulation 4, LR 1e-5.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 \
bash examples/simBenchmarks/LIBERO/train_files/run_u0_fast_libero.sh
```

Trailing arguments override YAML fields, including `--datasets.vla_data.data_root_dir`
and `--framework.u0.base_vlm`. `CONFIG`, `RUN_ID`, `MAX_TRAIN_STEPS`,
`SAVE_INTERVAL`, and `GRAD_ACCUM_STEPS` are also environment overrides.

Checkpoints contain the entire trained UNIS state and action-token mapping,
plus adjacent configuration and dataset normalization statistics. The frozen
IBQ weights are reloaded from their configured path. Loading rejects missing
backbone tensors, extra tensors, and incompatible action mappings. These model
checkpoints restore policy weights; they are not optimizer/scheduler resume
snapshots.

IBQ is deliberately kept outside the distributed parameter tree, since its
FP32 weights must not enter ZeRO-3's BF16 parameter collectives. The U0 interface
explicitly handles its device moves and keeps it frozen/eval.

## Evaluation

```bash
GPU_IDS=0 \
CKPT="$PWD/playground/Checkpoints/u0_fast_libero_goal_smoke/checkpoints/steps_10_model.safetensors" \
MAX_TASKS=1 NUM_TRIALS_PER_TASK=1 \
bash examples/simBenchmarks/LIBERO/eval_files/run_u0_fast_libero_eval.sh
```

The wrapper starts the existing WebSocket policy server, restores normalization,
runs LIBERO, and cleans up its own server. Logs go under the checkpoint run's
`eval_logs/`; videos/results use the existing LIBERO output layout. For a
standalone server, use `run_u0_fast_policy_server.sh` with the same `CKPT` and
`GPU_IDS`. `PORT` defaults to 6694.

One trial after ten training steps verifies the pipeline, not policy quality.
Full evaluation is explicitly requested with `MAX_TASKS=-1 NUM_TRIALS_PER_TASK=50`.

## Checks

```bash
/root/nas/envs/starvla-u0/bin/python -m pytest tests/test_u0_fast.py tests/test_emu_fast.py -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /root/nas/envs/starvla-u0/bin/python tests/u0_gpu_preflight.py
```

The GPU preflight checks real weights, two-view data, eager/FlashAttention
agreement, finite gradients and the frozen IBQ contract. It writes
`playground/Checkpoints/u0_preflight.json`.

## Verified local result (2026-09-10)

- 29 tests passed, including loss/gradient equivalence, full checkpoint restore,
  generation cache, UTF-8/FAST shape constraints, and EmuFast regressions.
- The final two-H100 ZeRO-3 smoke completed 10 optimizer steps (global batch 8).
  Logged loss changed from 35.0125 to 14.2097. The 400-tensor checkpoint contains
  the entire UNIS backbone and action mapping; attention, action embedding and
  output-head weight updates were verified against the original checkpoint.
- Real-weight preflight measured 1.248% relative hidden-state L2 difference
  between eager and FlashAttention, with the same next-token prediction. IBQ
  image IDs were identical before and after the framework BF16 cast.
- Restoring `u0_fast_libero_goal_smoke/checkpoints/steps_10_model.safetensors`
  into a new server completed one LIBERO Goal trial: **0/1 successes (0%)**.
  The failed-task video contains the full rollout. This is pipeline validation;
  no formal training or full benchmark evaluation was performed.

The machine-readable report, logs, dependency snapshots and video paths are in
`playground/Checkpoints/u0_fast_libero_goal_smoke/validation.json`. The simulator
emitted EGL destructor warnings after reporting the completed trial; its process
exited with code 0 and both server and simulator released their GPUs.

## Offline LIBERO vision cache

`run_u0_fast_libero.sh` now prepares and verifies the frozen FP32 IBQ codebook grids
before launching training. It uses the configured GPU count to shard episodes;
completed episodes are reused after interruption. The source videos and parquet
files are unchanged. The default cache is under `playground/Datasets/U0_VISION_CACHE`,
partitioned by vision encoder file stamps and preprocessing recipe. A complete
manifest is published only after all episodes of that subset have been verified.

Cached training skips video decoding and does not load IBQ. It retains the original
mixture sampler, action normalization, FAST encoding, prompts and labels. The
current preset has deterministic image preprocessing and two current camera views;
the cache generator supports `libero_goal` and the four-subset `libero_all` preset. Changing
image augmentation requires a new cache contract. Inference still encodes live
images and lazily loads IBQ if the checkpoint config used cached training.

To prepare the cache separately:

```bash
cd /root/nas/code/starVLA
/opt/conda/envs/starvla/bin/python -m starVLA.dataloader.u0_vision_cache \
  --config examples/simBenchmarks/LIBERO/train_files/u0_fast_libero_goal.yaml \
  --workers 8
```

A missing/incompatible cache fails training; it never silently falls back to online
encoding. File stamps detect normal source/weight replacements (path, size, mtime);
this is not a content checksum. `--limit-episodes` is diagnostic and never publishes
a complete manifest. Each log includes both cache preparation and training output.

### Raw vision cache format (version 3)

Each episode stores an int32 array `[frames, cameras, grid_height, grid_width]`.
At 224x224 input resolution this is `[frames, 2, 14, 14]`, containing only IBQ
codebook IDs in `[0, 131072)`. No text-token IDs, size strings, row separators or
image-boundary markers are cached. `U0Fast.forward` formats the raw grids with the
current text tokenizer; the current U0 format produces 217 tokens per image.
Changing the text tokenizer or prompt does not invalidate the vision cache.
Encoder weights, image resolution and preprocessing remain part of its identity.

Existing verified version-2 caches can be migrated without GPU encoding, preserving
the old files:

```bash
/opt/conda/envs/starvla/bin/python -m starVLA.dataloader.u0_vision_cache \
  --config examples/simBenchmarks/LIBERO/train_files/u0_fast_libero_goal.yaml \
  --migrate-formatted-cache playground/Datasets/U0_VISION_CACHE/636c1ef69908ac93399f/libero_goal_no_noops_1.0.0_lerobot
```

### Four-subset LIBERO training

`examples/simBenchmarks/LIBERO/train_files/u0_fast_libero_all.yaml` combines object,
goal, spatial and LIBERO-10 with equal dataset sampling weights. Other training
hyperparameters match the goal preset. The cache preparation walks all four subsets;
it reuses completed goal caches and prepares missing subset caches before training.

```bash
CONFIG=examples/simBenchmarks/LIBERO/train_files/u0_fast_libero_all.yaml \
RUN_ID=u0_fast_libero_all_full \
bash examples/simBenchmarks/LIBERO/train_files/run_u0_fast_libero.sh
```

Set `RUN_ID` as shown: the shared launcher otherwise overrides the YAML run ID with
its goal default.
# Eight-GPU Goal evaluation

From the starVLA root, run:

```bash
bash examples/simBenchmarks/LIBERO/eval_files/run_u0_fast_libero_goal_8gpu.sh \
  --checkpoint playground/Checkpoints/u0_fast_libero_all_zero2_full/checkpoints/steps_7500_model.safetensors
```

This first runs Goal task 0 on five fixed initial states, then starts a separate
10-task × 50-trial evaluation. Defaults: GPUs 0–7, ports 6800–6807, seed 7,
10 settling steps and at most 300 policy steps. `--gpus`, `--task-start`,
`--task-end`, `--trials`, `--base-port` and `--output` override the defaults.
Each GPU owns a policy server; `(task * trials + trial) % 8` assigns disjoint
episodes. U0 online FP32 IBQ performs bicubic resizing of the rotated raw camera
images. Inputs are two cameras plus instruction; action chunks are eight steps.

Outputs default to `checkpoints/eval/<checkpoint-stem>/libero_goal_<timestamp>/`
beside the checkpoint, with separate `smoke/` and `formal/` directories. Each
contains per-worker videos, JSONL episode records and logs; `results.json`
requires exact episode coverage and zero errors/invalid outputs for `valid=true`.
Root `provenance.json` records weight/config/encoder/source hashes. The launcher
refuses occupied GPUs/ports or an existing output directory and cleans up only
its own worker process groups on failure.

### Stateful h=2 interleaved training

Run `bash examples/simBenchmarks/LIBERO/train_files/run_u0_fast_libero_all_zero2_h2.sh` from any directory. This starts a new U0-4B run named `u0_fast_libero_all_zero2_h2_20k`: 20,000 optimizer updates, saves at 5k/10k/15k/20k. It does not load the existing step-7500 policy.

Each example contains instruction + current 8D state + two current camera grids + FAST actions t:t+8 + two camera grids at t+8 + state at t+8 + FAST actions t+8:t+16. The state is position (3), axis-angle (3), and both gripper joint positions (2). State statistics are computed from the raw frames of all selected training trajectories, quantized to 256 bins; the legacy `pad` field is retained. Existing starVLA action normalization is unchanged. Incomplete final windows are excluded by the mixture sampler itself. Raw per-frame IBQ caches are reused.

Both action segments and their end markers, the final EOS, and only the second pair's 392 visual code tokens are supervised. Loss is mean action/end CE plus mean future-image CE, with equal group weights. Instruction/state/initial image/format tokens remain masked. Sequence length is capped at 2048 without truncation. Logs include action CE, future-image CE, action accuracy, token counts and GPU peak memory.

`MAX_TRAIN_STEPS`, `SAVE_INTERVAL`, `RUN_ID`, `PER_DEVICE_BATCH_SIZE` and `GRAD_ACCUM_STEPS` override launch defaults. State/encoder identity and window manifests are written under `<run>/preparation/`, embedded in the saved model config, and the sequence contract hash is saved in checkpoint metadata. These are model-only safetensors checkpoints: optimizer/RNG/data progress are not saved.

The existing Goal evaluation shell accepts the new checkpoint via `--checkpoint`. Its server handshake enables raw state input and bounded history automatically. The client executes each eight-action chunk, then sends the previous actual observation plus the exact returned FAST tokens together with the current observation. Reset/task change/error clears history; no predicted future image is used in simulation. Bicubic RGB preprocessing and the 180-degree simulator image rotation remain shared with the previous U0 evaluation.

The h=2 launch defaults are **4 examples/GPU × 8 GPUs × accumulation 8 = global batch 256**. The requested 8/GPU × accumulation 4 ran out of memory during real backward passes; the h=1 capacity result does not apply to future-image supervision. To change batch, set both `PER_DEVICE_BATCH_SIZE` and `GRAD_ACCUM_STEPS` deliberately.
