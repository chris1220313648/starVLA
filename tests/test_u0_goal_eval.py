import json
import os
import signal
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from examples.simBenchmarks.LIBERO.eval_files.eval_libero import assigned_trials, _binarize_gripper_open
from examples.simBenchmarks.LIBERO.eval_files.u0_fast_eval_control import aggregate
from examples.simBenchmarks.LIBERO.eval_files import model2libero_interface as interface


class EvaluationTests(unittest.TestCase):
    def test_worker_exits_and_cleans_server(self):
        script=Path('examples/simBenchmarks/LIBERO/eval_files/run_u0_fast_libero_goal_8gpu.sh').read_text()
        function=script[script.index('worker() {'):script.index('export -f worker')]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fake=root/'python'
            fake.write_text('#!/usr/bin/env python3\nimport os,sys,time\nfrom pathlib import Path\n'
                'assert not any(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"))\n'
                'if any(arg.endswith("server_policy.py") for arg in sys.argv[1:]):\n'
                ' Path(os.environ["PHASE_DIR"],"server_pid").write_text(str(os.getpid()))\n'
                ' time.sleep(120)\n')
            fake.chmod(0o755)
            env={**os.environ,'PHASE_DIR':tmp,'BASE_PORT':'6800','EVAL_DIR':tmp,
                 'LIBERO_PYTHON':str(fake),'STARVLA_PYTHON':str(fake),'CONTROL':'control',
                 'CKPT':'checkpoint','PHASE_START':'0','PHASE_END':'0','PHASE_TRIALS':'5'}
            env.update(RANK='0', WORLD_SIZE='8', LOCAL_RANK='0', LOCAL_WORLD_SIZE='8',
                       GROUP_RANK='0', ROLE_RANK='0', ROLE_WORLD_SIZE='8',
                       MASTER_ADDR='127.0.0.1', MASTER_PORT='23456')
            proc=subprocess.Popen(['bash','-euo','pipefail','-c',function+'\nworker 0 0'],env=env,
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            try:
                _,err=proc.communicate(timeout=8)
                self.assertEqual(proc.returncode,0,err.decode()+(root/'worker_0/server.log').read_text())
                pid=int((root/'server_pid').read_text())
                with self.assertRaises(ProcessLookupError): os.kill(pid,0)
            finally:
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()

    def test_exact_500_assignment(self):
        groups = [[(t,i) for t in range(10) for i in assigned_trials(t,50,r,8)] for r in range(8)]
        flattened = [pair for group in groups for pair in group]
        self.assertEqual(len(flattened),500)
        self.assertEqual(len(set(flattened)),500)
        self.assertEqual(sorted(map(len,groups)),[62]*4+[63]*4)
        self.assertEqual(sum(len(assigned_trials(0,5,r,8)) for r in range(8)),5)

    def test_raw_camera_chunk_reset_and_invalid(self):
        class FakeServer:
            def __init__(self,*args): self.calls=[]; self.bad=False
            def get_server_metadata(self): return dict(action_chunk_size=8)
            def predict_action(self,query):
                self.calls.append(query)
                actions=np.zeros((1,8,7),dtype=np.float32)
                if self.bad: actions[0,0,0]=np.nan
                return {'data':{'actions':actions}}
        with patch.object(interface,'WebsocketClientPolicy',FakeServer):
            client=interface.ModelClient(image_size=None)
            image=np.arange(256*256*3,dtype=np.uint8).reshape(256,256,3)
            example={'image':[image,image.copy()],'lang':'task'}
            client.reset('task')
            for step in range(10): client.step(example,step)
            self.assertEqual(len(client.client.calls),2)
            np.testing.assert_array_equal(client.client.calls[0]['examples'][0]['image'][0],image)
            self.assertNotIn('state',client.client.calls[0]['examples'][0])
            client.reset('task'); client.step(example,0)
            self.assertEqual(len(client.client.calls),3)
            client.client.bad=True; client.reset('task')
            with self.assertRaises(ValueError): client.step(example,0)
        np.testing.assert_array_equal(_binarize_gripper_open(1),[-1])
        np.testing.assert_array_equal(_binarize_gripper_open(0),[1])

    def test_aggregate_rejects_duplicates_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(5):
                p=root/f'worker_{i}'; p.mkdir()
                row=dict(task=0,trial=i,success=i==0,error=None,invalid_output=False)
                (p/'episodes.jsonl').write_text(json.dumps(row)+'\n')
                (p/'results.json').write_text(json.dumps(dict(complete=True)))
            self.assertTrue(aggregate(root,0,0,5))
            self.assertEqual(json.loads((root/'results.json').read_text())['success_rate'],.2)
            p=root/'worker_4/episodes.jsonl'
            row['trial']=0; p.write_text(json.dumps(row)+'\n')
            self.assertFalse(aggregate(root,0,0,5))
            row.update(trial=4,error='invalid action',invalid_output=True)
            p.write_text(json.dumps(row)+'\n')
            self.assertFalse(aggregate(root,0,0,5))


if __name__=='__main__': unittest.main()
