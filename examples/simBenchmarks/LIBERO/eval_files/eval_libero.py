import dataclasses
import json
import logging
import math
import os
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"
    task_start: int = 0
    task_end: int = -1
    worker_id: int = 0
    num_workers: int = 1
    u0_fast: bool = False
    u0_oat: bool = False
    execution_steps: int = 0  # Actions executed from each prediction; 0 uses the full chunk.
    render_check: bool = False


def assigned_trials(task_id, trials, worker_id, num_workers):
    if trials <= 0 or num_workers <= 0 or not 0 <= worker_id < num_workers:
        raise ValueError("Invalid trial count or worker assignment")
    return [i for i in range(trials) if (task_id * trials + i) % num_workers == worker_id]


def eval_libero(args: Args) -> None:
    logging.info("Arguments: %s", json.dumps(dataclasses.asdict(args)))
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    limits = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520, libero_90=400)
    max_steps = limits[args.task_suite_name]
    end = suite.n_tasks - 1 if args.task_end < 0 else args.task_end
    if args.max_tasks > 0:
        end = min(end, args.max_tasks - 1)
    if not 0 <= args.task_start <= end < suite.n_tasks:
        raise ValueError("Invalid task range")
    assigned_trials(args.task_start, args.num_trials_per_task, args.worker_id, args.num_workers)
    output = pathlib.Path(args.video_out_path)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "episodes.jsonl"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {result_path}")
    (output / "args.json").write_text(json.dumps(dataclasses.asdict(args), indent=2))
    if args.render_check:
        env, _ = _get_libero_env(suite.get_task(args.task_start), LIBERO_ENV_RESOLUTION, args.seed)
        try:
            obs = env.reset()
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                assert obs[key].shape == (256, 256, 3), (key, obs[key].shape)
            print("RENDER_OK", flush=True)
        finally:
            env.close()
        return
    # U0 owns RGB/bicubic resizing, matching its offline IBQ cache recipe.
    client = ModelClient(host=args.host, port=args.port, unnorm_key=args.unnorm_key,
                         image_size=None if (args.u0_fast or args.u0_oat) else (224, 224),
                         execution_horizon=args.execution_steps or None)
    meta = client._server_metadata
    expected_horizon = 32 if meta.get('action_codec') == 'oat' else 8
    if args.u0_oat and meta.get('action_codec') != 'oat':
        client.client.close()
        raise ValueError('Expected an OAT policy')
    if (args.u0_fast or args.u0_oat) and (client.action_chunk_size != expected_horizon or
            pathlib.Path(meta["ckpt_path"]).resolve() != pathlib.Path(args.pretrained_path).resolve()):
        client.client.close()
        raise ValueError("Wrong policy checkpoint or action horizon")
    (output / "server_metadata.json").write_text(json.dumps(meta, indent=2))
    results = []
    started = time.monotonic()
    complete = False
    try:
        for task_id in range(args.task_start, end + 1):
            trials = assigned_trials(task_id, args.num_trials_per_task, args.worker_id, args.num_workers)
            if not trials:
                continue
            task = suite.get_task(task_id)
            states = suite.get_task_init_states(task_id)
            if max(trials) >= len(states):
                raise ValueError("Not enough fixed initial states")
            env, description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            try:
                for trial in trials:
                    row = dict(task=task_id, trial=trial, success=False, error=None,
                               invalid_output=False, steps=0, policy_calls=0)
                    tick = time.monotonic()
                    frames = []
                    try:
                        client.reset(task_description=description)
                        env.reset()
                        obs = env.set_init_state(states[trial])
                        for _ in range(args.num_steps_wait):
                            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                            if done:
                                raise RuntimeError("Environment terminated during settling")
                        for step in range(max_steps):
                            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                            frames.append(image)
                            try:
                                example = {"image": [image, wrist], "lang": description}
                                if client._server_metadata.get('requires_raw_state', False):
                                    example['state'] = np.r_[obs['robot0_eef_pos'], _quat2axisangle(np.array(obs['robot0_eef_quat'], copy=True)), obs['robot0_gripper_qpos']].astype(np.float32)
                                response = client.step(example, step=step)
                                raw = response["raw_action"]
                                xyz = np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1)
                                rot = np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1)
                                grip = np.asarray(raw["open_gripper"], dtype=np.float32).reshape(-1)
                                if (xyz.size, rot.size, grip.size) != (3, 3, 1) or not np.isfinite(np.r_[xyz, rot, grip]).all():
                                    raise ValueError("Invalid action shape or nonfinite values")
                            except Exception as exc:
                                row["invalid_output"] = isinstance(exc, ValueError) or any(
                                    token in str(exc).lower() for token in ("fast sequence", "malformed", "unterminated", "invalid actions", "coefficients"))
                                raise
                            row["policy_calls"] += int(step % client.execution_horizon == 0)
                            action = np.r_[xyz, rot, _binarize_gripper_open(grip)]
                            obs, _, done, _ = env.step(action.tolist())
                            row["steps"] += 1
                            if done:
                                row["success"] = True
                                break
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    finally:
                        try:
                            if frames:
                                video = output / f"task_{task_id}_trial_{trial}_{'success' if row['success'] else 'failure'}.mp4"
                                imageio.mimwrite(video, frames, fps=10)
                                row["video"] = str(video)
                        except Exception as exc:
                            row["error"] = f"Video write failed: {exc}"
                            raise
                        finally:
                            row["seconds"] = time.monotonic() - tick
                            results.append(row)
                            with result_path.open("a") as f:
                                f.write(json.dumps(row) + "\n")
                            print(json.dumps(row), flush=True)
            finally:
                env.close()
        complete = True
    finally:
        client.client.close()
        report = dict(complete=complete, episodes=len(results), successes=sum(r["success"] for r in results),
                      errors=sum(r["error"] is not None for r in results),
                      invalid_outputs=sum(r["invalid_output"] for r in results), seconds=time.monotonic()-started)
        (output / "results.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
