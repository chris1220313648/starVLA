"""Check one cached scene in each RoboTwin suite without loading a policy.

Run with ../../envs/roboscope-kit/bin/python from the starVLA repository root.
"""

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

from evaluate_groundingpi_cached import KIT, ROOT, SUITES, WORKSPACE


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def check_observation(observation, output, stage):
    import numpy as np
    from PIL import Image

    cameras = {}
    for name in ("head_camera", "left_camera", "right_camera"):
        rgb = np.asarray(observation["observation"][name]["rgb"])
        if (rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1
                or rgb.dtype != np.uint8 or np.ptp(rgb) == 0):
            raise ValueError(f"Invalid RGB image: {name}, {rgb.shape}, {rgb.dtype}")
        Image.fromarray(rgb).save(output / f"{stage}-{name}.png")
        cameras[name] = {"shape": list(rgb.shape), "std": float(rgb.std())}
    state = np.asarray(observation["joint_action"]["vector"])
    if state.ndim != 1 or state.size == 0 or not np.isfinite(state).all():
        raise ValueError("Robot joint state is empty or non-finite")
    for name, value in observation["endpose"].items():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"Non-finite end pose: {name}")
    return {"cameras": cameras, "joint_state": state.tolist()}


def check_scene(args):
    sys.path.insert(0, str(KIT / "benchmarks/robotwin"))
    from roboscope_robotwin_seed_scan import build_task_args, configure_cpu_threads

    configure_cpu_threads(1)
    config = build_task_args(WORKSPACE / "simulation/RoboTwin", args.task, args.suite)
    # Keep the same cached EE setup path, without loading a learned policy.
    from roboscope_robotwin_launcher import install_ee_eval_planner

    install_ee_eval_planner()
    with (args.cache_source / "seed-cache" / args.suite / f"{args.task}.pkl").open("rb") as handle:
        entry = pickle.load(handle)[0]
    task = getattr(importlib.import_module(f"envs.{args.task}"), args.task)()
    result = {"suite": args.suite, "task": args.task, "seed": entry["seed"],
              "physics_steps": 10, "passed": False}
    started = time.monotonic()
    try:
        task.setup_demo(now_ep_num=0, seed=entry["seed"], is_test=True, **config)
        result["before"] = check_observation(task.get_obs(), args.output, "before")
        for _ in range(10):
            task.scene.step()
        result["after"] = check_observation(task.get_obs(), args.output, "after")
        result["passed"] = True
    finally:
        try:
            if hasattr(task, "scene"):
                task.close_env()
        finally:
            result["elapsed_seconds"] = round(time.monotonic() - started, 2)
            write_json(args.output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-source", type=Path, default=WORKSPACE / "experiments/roboscope-groundingpi/cache_20261110_170840")
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, default=Path("/tmp") / datetime.now(timezone.utc).strftime("robotwin_scene_check_%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--suite", choices=SUITES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.cache_source = args.cache_source.resolve()
    args.output = args.output.resolve()
    if args.suite:
        check_scene(args)
        return

    sys.path[:0] = [str(KIT / "packages/roboscope-kit/src"), str(KIT / "examples/groundingpi")]
    from evaluate_local import run, revision
    from evaluate_groundingpi_cached import verify_cache
    from roboscope_kit.runtime.benchmarks.robotwin.adapter import ROBOTWIN_ALL_TASKS
    from roboscope_kit.runtime.benchmarks.robotwin.seed_cache import validate_seed_cache_dir

    if args.task not in ROBOTWIN_ALL_TASKS:
        parser.error(f"Unknown RoboTwin task: {args.task}")
    if not args.gpu.isdigit():
        parser.error("--gpu must be a single integer GPU ID")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"passed": False, "cache_source": str(args.cache_source),
               "task": args.task, "gpu": args.gpu, "suites": {}}
    print(f"Output: {args.output}", flush=True)
    try:
        hashes = verify_cache(args.cache_source, ROBOTWIN_ALL_TASKS,
                              revision(WORKSPACE / "simulation/RoboTwin"))
        for suite in SUITES:
            validate_seed_cache_dir(args.cache_source / "seed-cache", suite, ROBOTWIN_ALL_TASKS, 20)
        summary["cache_validation"] = {"passed": True, "cache_files": len(hashes)}
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, SIM_SAVE_VIDEO="0",
                   PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
        driver = args.cache_source / "driver-libs/usr"
        if (driver / "share/vulkan/icd.d/nvidia_icd.json").is_file():
            env["VK_ICD_FILENAMES"] = str(driver / "share/vulkan/icd.d/nvidia_icd.json")
            env["LD_LIBRARY_PATH"] = str(driver / "lib/x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", "")
        for suite in SUITES:
            output = args.output / suite
            output.mkdir()
            command = [str(WORKSPACE / "envs/robotwin/bin/python"), str(Path(__file__).resolve()),
                       "--suite", suite, "--task", args.task, "--cache-source", str(args.cache_source),
                       "--output", str(output)]
            try:
                run(command, output / "scene.log", env, ROOT, 300)
                summary["suites"][suite] = json.loads((output / "result.json").read_text())
            except Exception as exc:
                summary["suites"][suite] = {"passed": False, "error": str(exc), "log": str(output / "scene.log")}
            print(f"{suite}: {'PASS' if summary['suites'][suite]['passed'] else 'FAIL'}", flush=True)
        summary["passed"] = all(item["passed"] for item in summary["suites"].values())
    except Exception as exc:
        summary["error"] = str(exc)
        traceback.print_exc()
    finally:
        write_json(args.output / "summary.json", summary)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
