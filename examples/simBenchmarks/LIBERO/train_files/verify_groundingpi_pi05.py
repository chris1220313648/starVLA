"""Strict reload, real-data prediction and cached/full attention verification.

Run on one free GPU after the four-rank training job exits. Writes verification.json
and verification_actions.npy beside config.full.yaml; failures exit nonzero.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import baseframework


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = OmegaConf.load(run / 'config.full.yaml')
    serving_cfg = OmegaConf.load(run / 'config.yaml')
    assert serving_cfg.datasets.vla_data.include_state
    assert serving_cfg.datasets.vla_data.normalize_state_on_server
    dataset = get_vla_dataset(cfg.datasets.vla_data)
    samples = [part[0] for part in dataset.datasets]
    checkpoint = run / 'final_model/pytorch_model.pt'
    noise = np.random.default_rng(7).standard_normal((len(samples), 16, 7)).astype(np.float32)
    model = baseframework.from_pretrained(checkpoint).cuda().eval()
    reference = model.predict_action(samples, noise=noise)['normalized_actions']
    assert reference.shape == noise.shape and np.isfinite(reference).all()
    # Compare identical full/cached paths in FP32 expert/BF16 backbone arithmetic.
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        prefix, mask = model.prepare_prefix(samples[:1])
        x = torch.as_tensor(noise[:1], device='cuda')
        time = torch.tensor([.5], device='cuda')
        full = model.velocity(prefix, mask, x, time)
        cache = model.prefix_cache(prefix, mask)
        shapes = [(k.shape, v.shape) for k, v in cache]
        cached = model.velocity(None, mask, x, time, cache)
        torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)
        model.velocity(None, mask, x, time / 2, cache)
        assert shapes == [(k.shape, v.shape) for k, v in cache]
        cache_error = float((full - cached).abs().max())
    # Keep batch shape identical so a batch-size-dependent kernel choice cannot
    # be mistaken for sensitivity to the modified conditioning input.
    single_reference = model.predict_action(samples[:1], noise=noise[:1])['normalized_actions']
    changes = {}
    for kind in ('image', 'lang', 'state'):
        altered = copy.deepcopy(samples[:1])
        if kind == 'image':
            altered[0]['image'] = [Image.new('RGB', image.size) for image in altered[0]['image']]
        elif kind == 'lang':
            altered[0]['lang'] = 'do nothing and keep the robot still'
        else:
            altered[0]['state'] = np.zeros_like(altered[0]['state'])
        prediction = model.predict_action(altered, noise=noise[:1])['normalized_actions']
        changes[kind] = float(np.max(np.abs(prediction - single_reference)))
        assert changes[kind] > 1e-5, f'Prediction ignored {kind}'
    del model, prefix, mask, cache, full, cached, x, time
    gc.collect()
    torch.cuda.empty_cache()
    restored = baseframework.from_pretrained(checkpoint).cuda().eval()
    actual = restored.predict_action(samples, noise=noise)['normalized_actions']
    np.testing.assert_allclose(reference, actual, atol=1e-5, rtol=1e-5)
    health = [json.loads(row) for row in (run / 'health.jsonl').read_text().splitlines()]
    last = health[-1]
    assert last['step'] == 100
    assert len(last['ranks']) == 4
    for rank in last['ranks']:
        assert not rank['stop']
        assert all(rank['representative_parameters_updated'].values()), rank
        assert all(value > 0 and np.isfinite(value) for value in rank['representative_grad_norms'].values()), rank
    metrics = [json.loads(row) for row in (run / 'metrics.jsonl').read_text().splitlines()]
    assert all(np.isfinite(row['action_dit_loss']) for row in metrics)
    result = dict(status='passed', checkpoint=str(checkpoint), strict_reload=True,
                  samples=len(samples), output_shape=list(actual.shape),
                  reload_max_abs_error=float(np.max(np.abs(reference-actual))),
                  cached_full_max_abs_error=cache_error, condition_max_abs_change=changes,
                  final_health=last, first_metrics=metrics[0], last_metrics=metrics[-1])
    np.save(run / 'verification_actions.npy', actual)
    (run / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
