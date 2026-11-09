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
    eef = cfg.datasets.vla_data.dataset_py in ('robotwin_eef', 'robotwin_eef_lerobot')
    separate = eef and getattr(cfg.datasets.vla_data, 'image_input_mode', 'composite') == 'separate'
    if eef:
        from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_dataset import get_vla_dataset as get_eef_dataset
        if cfg.datasets.vla_data.dataset_py == 'robotwin_eef_lerobot':
            from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_lerobot_dataset import get_vla_dataset as get_eef_dataset
        dataset = get_eef_dataset(cfg.datasets.vla_data, training=False)
        samples = [dataset[0], dataset[len(dataset)//2]]
    else:
        dataset = get_vla_dataset(cfg.datasets.vla_data)
        samples = [part[0] for part in dataset.datasets]
    checkpoint = run / 'final_model/pytorch_model.pt'
    action_cfg = cfg.framework.action_model
    expected_dim = 20 if eef else 14
    assert action_cfg.state_dim == expected_dim and action_cfg.action_dim == expected_dim
    expected_image_count = 3 if separate or not eef else 1
    expected_image_size = (320,240) if separate else (320,384) if eef else (224,224)
    assert all(len(s['image']) == expected_image_count and
               all(im.size == expected_image_size for im in s['image']) for s in samples)
    noise = np.random.default_rng(7).standard_normal((len(samples), int(action_cfg.action_horizon), int(action_cfg.action_dim))).astype(np.float32)
    model = baseframework.from_pretrained(checkpoint).cuda().eval()
    reference = model.predict_action(samples, noise=noise)['normalized_actions']
    assert reference.shape == noise.shape and np.isfinite(reference).all()
    # Compare identical full/cached paths in FP32 expert/BF16 backbone arithmetic.
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        prefix, mask = model.prepare_prefix(samples[:1])
        x = torch.as_tensor(noise[:1], device='cuda')
        time = torch.tensor([.5], device='cuda')
        state = model._continuous_state(samples[:1])
        full = model.velocity(prefix, mask, x, time, state=state)
        cache = model.prefix_cache(prefix, mask)
        shapes = [(k.shape, v.shape) for k, v in cache]
        cached = model.velocity(None, mask, x, time, cache, state=state)
        torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)
        model.velocity(None, mask, x, time / 2, cache, state=state)
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
    assert last['step'] == int(cfg.trainer.max_train_steps)
    expected_ranks = OmegaConf.load(run / 'accelerate.yaml').num_processes
    assert len(last['ranks']) == expected_ranks
    for rank in last['ranks']:
        assert not rank['stop']
        assert all(rank['representative_parameters_updated'].values()), rank
        assert all(value > 0 and np.isfinite(value) for value in rank['representative_grad_norms'].values()), rank
    isolation = None
    if restored.knowledge_insulation:
        restored.train()
        isolation = {}
        for loss_name in ('action_loss', 'fast_token_loss'):
            restored.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = restored(samples[:1])
            assert all(torch.isfinite(value) for value in losses.values())
            losses[loss_name].backward()
            backbone_has_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in restored.qwen_vl_interface.parameters())
            expert_has_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in restored.action_model.parameters())
            if loss_name == 'action_loss':
                assert not backbone_has_grad and expert_has_grad
                assert all(p.grad is None for p in restored.qwen_vl_interface.parameters())
            else:
                assert backbone_has_grad and not expert_has_grad
                assert all(p.grad is None for p in restored.action_model.parameters())
            isolation[loss_name] = dict(backbone_has_grad=bool(backbone_has_grad), expert_has_grad=bool(expert_has_grad))
            del losses
        restored.zero_grad(set_to_none=True)
    cot_gradients = None
    if restored.cot_loss_weight:
        labelled = next((sample for sample in samples if sample.get('cot')), None)
        if labelled is None:
            labelled = next((row for i in range(len(dataset)) if (row := dataset[i]).get('cot')), None)
        if labelled is None:
            raise ValueError('CoT verification requires at least one labelled frame')
        restored.train()
        restored.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = restored([labelled])
            loss, count = output['cot_loss'], output['cot_samples']
        assert count == 1 and torch.isfinite(loss) and loss > 0
        loss.backward()
        head_grad = restored.qwen_vl_interface.model.lm_head.weight.grad
        backbone_has_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in restored.language_model.parameters())
        assert head_grad is not None and head_grad.abs().max() > 0 and backbone_has_grad
        assert all(p.grad is None for p in restored.action_model.parameters())
        cot_gradients = dict(backbone_has_grad=True, language_head_has_grad=True, expert_has_grad=False)
        restored.zero_grad(set_to_none=True)
    metrics = [json.loads(row) for row in (run / 'metrics.jsonl').read_text().splitlines()]
    assert all(np.isfinite(row['action_dit_loss']) for row in metrics)
    if restored.cot_loss_weight:
        assert all(np.isfinite(row['cot_loss']) and np.isfinite(row['total_loss']) for row in metrics)
    if restored.knowledge_insulation:
        assert all(np.isfinite(row['fast_token_loss']) and np.isfinite(row['total_loss']) for row in metrics)
    result = dict(status='passed', checkpoint=str(checkpoint), strict_reload=True,
                  samples=len(samples), output_shape=list(actual.shape), knowledge_insulation_gradients=isolation,
                  cot_gradients=cot_gradients,
                  reload_max_abs_error=float(np.max(np.abs(reference-actual))),
                  cached_full_max_abs_error=cache_error, condition_max_abs_change=changes,
                  final_health=last, first_metrics=metrics[0], last_metrics=metrics[-1])
    np.save(run / 'verification_actions.npy', actual)
    (run / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
