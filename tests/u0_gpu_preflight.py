"""Explicit GPU integration check: run from the repository root."""

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.VLM4A.U0Fast import U0Fast
from starVLA.model.modules.vlm.u0.modeling_unis import UNISAttention, UNISFlashAttention2


def main():
    torch.manual_seed(42)
    cfg = OmegaConf.load("examples/simBenchmarks/LIBERO/train_files/u0_fast_libero_goal.yaml")
    model = U0Fast(cfg).cuda().eval()
    interface = model.u0_interface
    sample = get_vla_dataset(cfg.datasets.vla_data)[0]
    images, instructions = model._observations([sample])
    image_ids = interface.encode_image(images[0][0])
    model.bfloat16()
    assert image_ids == interface.encode_image(images[0][0])
    assert next(interface.vq_model.parameters()).dtype == torch.float32
    tokens = model.action_model.encoder_action2fastoken([sample["action"]])
    inputs = interface.build_training_inputs(images, instructions, tokens)
    decoder = interface.model.get_decoder()
    with torch.no_grad():
        flash = decoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False).last_hidden_state
        for layer in decoder.layers:
            layer.self_attn.__class__ = UNISAttention
        decoder._use_flash_attention_2 = False
        eager = decoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False).last_hidden_state
        # Compare aggregate BF16 numerical error and next-token predictions.
        relative_error = (flash.float() - eager.float()).norm() / eager.float().norm()
        torch.testing.assert_close(relative_error, torch.zeros_like(relative_error), atol=0.03, rtol=0)
        flash_next = interface.model.lm_head(flash[:, -1]).argmax(-1)
        eager_next = interface.model.lm_head(eager[:, -1]).argmax(-1)
        assert torch.equal(flash_next, eager_next)
        for layer in decoder.layers:
            layer.self_attn.__class__ = UNISFlashAttention2
        decoder._use_flash_attention_2 = True
    del flash, eager
    model.train()
    loss = interface.action_loss(inputs)
    loss.backward()
    assert all(p.requires_grad for p in interface.model.parameters())
    assert all(p.grad is None for p in interface.vq_model.parameters())
    assert not interface.vq_model.training
    gradient = decoder.layers[0].self_attn.q_proj.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    report = {
        "checkpoint": cfg.framework.u0.base_vlm,
        "loss": loss.item(), "flash_eager_relative_l2": relative_error.item(),
        "action_shape": list(sample["action"].shape), "views": len(images[0]),
        "sequence_length": inputs["input_ids"].shape[1], "fast_token_count": len(tokens[0]),
        "trainable_parameters": sum(p.numel() for p in interface.model.parameters()),
        "q_proj_grad_norm": gradient.float().norm().item(),
        "ibq_tokens_preserved_after_bfloat16": True,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    path = Path("playground/Checkpoints/u0_preflight.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
