"""Real-weight training/inference smoke. Run from repository root with python -m ... ."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.VLM4A.GroundingPIPI import GroundingPIPI
from starVLA.model.framework.VLM4A.GroundingPIOFT import GroundingPIOFT
from starVLA.model.framework.VLM4A.GroundingPIFast import GroundingPIFast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['GroundingPIPI', 'GroundingPIOFT', 'GroundingPIFast'], required=True)
    parser.add_argument('--output', default='playground/groundingpi-smoke')
    parser.add_argument('--fit-steps', type=int, default=60)
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    output = Path(args.output) / args.framework
    output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(f'examples/modelExtensions/GroundingPI/train_files/{args.framework.lower()}.yaml')
    start = time.time()
    cls = dict(GroundingPIPI=GroundingPIPI, GroundingPIOFT=GroundingPIOFT, GroundingPIFast=GroundingPIFast)[args.framework]
    model = cls(cfg).cuda()
    image = Image.fromarray(np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8))
    batch = [dict(image=[image], lang='Move.', action=np.zeros((16, 7), np.float32), state=np.zeros((1, 7), np.float32)),
             dict(image=[image, image], lang='Move the red block to the left.', action=np.zeros((16, 7), np.float32), state=np.zeros((1, 7), np.float32))]
    model.train()
    loss = model(batch)['action_loss']
    assert torch.isfinite(loss), loss
    loss.backward()
    grads = {name: False for name in ['qwen_vl_interface', 'action_model']}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            if parameter.grad.abs().max() > 0:
                grads[name.split('.')[0]] = True
    assert grads['qwen_vl_interface'], grads
    if args.framework != 'GroundingPIFast':
        assert grads['action_model'], grads
    print('FORWARD_BACKWARD', float(loss), grads, flush=True)
    initial_loss = float(loss.detach())
    del loss
    model.zero_grad(set_to_none=True)
    if args.framework == 'GroundingPIFast':
        # Teach newly initialized action rows one synthetic trajectory. Full-backbone
        # gradients were checked above; fitting only the output head keeps this small.
        model.requires_grad_(False)
        head = model.qwen_vl_interface.model.get_output_embeddings()
        head.requires_grad_(True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3)
        model.eval()
        for step in range(args.fit_steps):
            optimizer.zero_grad(set_to_none=True)
            fit_loss = model(batch)['action_loss']
            fit_loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print('FIT', step, float(fit_loss), flush=True)
        del optimizer
        model.zero_grad(set_to_none=True)
        tokenizer = model.action_model.fast_tokenizer
        decoded = np.asarray(tokenizer.decode(tokenizer(np.stack([e['action'] for e in batch]))))
        assert decoded.shape == (2, 16, 7) and np.isfinite(decoded).all()
    model.eval()
    prediction = model.predict_action(batch)['normalized_actions']
    assert prediction.shape == (2, 16, 7) and np.isfinite(prediction).all()
    torch.manual_seed(123)
    single = model.predict_action(batch[:1])['normalized_actions']
    assert single.shape == (1, 16, 7) and np.isfinite(single).all()
    print('INFERENCE', prediction.shape, flush=True)
    checkpoint = output / 'model.pt'
    torch.save(model.state_dict(), checkpoint)
    OmegaConf.save(model.config, output / 'config.yaml')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    restored = cls(OmegaConf.load(output / 'config.yaml')).cuda()
    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    restored.eval()
    torch.manual_seed(123)
    result = restored.predict_action(batch[:1])['normalized_actions']
    assert result.shape == (1, 16, 7) and np.isfinite(result).all()
    np.testing.assert_allclose(result, single, rtol=1e-4, atol=1e-5)
    report = dict(framework=args.framework, loss=initial_loss, gradients=grads,
                  prediction_shape=list(prediction.shape), reload=True,
                  peak_memory_gib=torch.cuda.max_memory_allocated()/2**30, seconds=time.time()-start)
    (output / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
