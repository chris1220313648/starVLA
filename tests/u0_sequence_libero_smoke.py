"""Bounded real LIBERO rollout against a stateful U0 websocket server."""
import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark
from examples.simBenchmarks.LIBERO.eval_files.eval_libero import _get_libero_env, _quat2axisangle, _binarize_gripper_open
from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,default=6818)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    suite=benchmark.get_benchmark_dict()['libero_goal']()
    env,description=_get_libero_env(suite.get_task(0),256,7)
    client=ModelClient(host='127.0.0.1',port=args.port,unnorm_key='franka',image_size=None,action_ensemble=False)
    assert client.sequence_h==2
    frames=[]
    started=time.monotonic()
    done=False
    try:
        env.reset()
        obs=env.set_init_state(suite.get_task_init_states(0)[0])
        client.reset(description)
        for _ in range(10):
            obs,_,done,_=env.step([0,0,0,0,0,0,-1])
            assert not done
        for t in range(2 * client.action_chunk_size):
            images=[np.ascontiguousarray(obs[k][::-1,::-1]) for k in ('agentview_image','robot0_eye_in_hand_image')]
            state=np.r_[obs['robot0_eef_pos'],_quat2axisangle(np.array(obs['robot0_eef_quat'],copy=True)),obs['robot0_gripper_qpos']]
            frames.append(images[0])
            raw=client.step(dict(image=images,state=state,lang=description),step=t)['raw_action']
            action=np.r_[raw['world_vector'],raw['rotation_delta'],_binarize_gripper_open(raw['open_gripper'])]
            assert action.shape==(7,) and np.isfinite(action).all() and action[-1] in (-1,1)
            obs,_,done,_=env.step(action.tolist())
            if done: break
        assert len(client.sequence_history)==1, 'Rollout did not exercise the second observation/action window'
        token_key = 'action_tokens' if client.action_codec == 'oat' else 'fast_tokens'
        history_tokens=client.sequence_history[0][token_key]
        if client.action_codec == 'oat':
            assert len(history_tokens) == 16 and all(0 <= token < 1920 for token in history_tokens)
        client.reset(description)
        assert not client.sequence_history and client.pending_history is None and client.raw_actions is None
        report=dict(task=0,initial_state=0,steps=len(frames),policy_calls=2,history_tokens=history_tokens,
                    action_codec=client.action_codec, action_horizon=client.action_chunk_size,
                    two_windows_exercised=True,reset_cleared=True,errors=0,invalid_outputs=0,
                    terminated=bool(done),seconds=time.monotonic()-started)
        imageio.mimsave(args.output/'rollout.mp4',frames,fps=10)
        (args.output/'results.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    finally:
        env.close()
        client.client._ws.close()


if __name__=='__main__': main()
