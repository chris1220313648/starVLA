# Copyright 2025 BAAI and 2026 ByteDance and/or its affiliates.
# Licensed under the Apache License, Version 2.0.

"""Emu3.5 adapter for StarVLA's image/instruction/action-token contract."""

from __future__ import annotations

import os.path as osp
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer

from .emu3_5 import Emu3Config, Emu3ForCausalLM
from .emu3_5.vision_tokenizer import build_vision_tokenizer


ACTION_TOKEN_COUNT = 2048
ACTION_TOKEN_START_ID = 149595


def _format_image_tokens(tokenizer, tokens: torch.Tensor) -> str:
    rows = []
    for row in tokens.tolist():
        rows.append("".join(f"<|visual token {token_id:0>6d}|>" for token_id in row))
    body = tokenizer.eol_token.join(rows)
    height, width = tokens.shape
    return f"{tokenizer.boi_token}{height}*{width}{tokenizer.img_token}{body}{tokenizer.eoi_token}"


class Emu3_5Interface(nn.Module):
    """Owns the Emu causal LM, IBQ encoder, tokenizer, and LoRA adapters."""

    def __init__(self, config):
        super().__init__()
        cfg = config.framework.emu
        self.image_size = int(cfg.get("image_size", 224))
        self.max_length = int(cfg.get("max_length", 2048))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 64))

        tokenizer_path = cfg.get(
            "tokenizer_path", osp.join(osp.dirname(__file__), "emu3_5", "tokenizer_emu3_ibq")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            special_tokens_file=osp.join(tokenizer_path, "emu3_vision_tokens.txt"),
            trust_remote_code=True,
        )
        self._set_special_tokens()

        model_cfg = Emu3Config.from_pretrained(cfg.base_vlm, trust_remote_code=True)
        load_kwargs = {
            "config": model_cfg,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": cfg.get("attn_implementation", "flash_attention_2"),
            "low_cpu_mem_usage": True,
        }
        device_map = cfg.get("device_map", None)
        if device_map not in (None, "none", "null", ""):
            load_kwargs["device_map"] = device_map
        # Reuse pretrained text rows; never resize the vocabulary for FAST actions.
        start = int(cfg.get("action_token_start_id", ACTION_TOKEN_START_ID))
        self.action_token_ids = list(range(start, start + ACTION_TOKEN_COUNT))
        text_ids = set(self.tokenizer.mergeable_ranks.values())
        if not set(self.action_token_ids).issubset(text_ids) or start < 0:
            raise ValueError("FAST action IDs must belong entirely to the existing text vocabulary")
        if len(self.tokenizer) != model_cfg.vocab_size:
            raise ValueError("Emu tokenizer and pretrained model vocabulary sizes differ")
        model = Emu3ForCausalLM.from_pretrained(cfg.base_vlm, **load_kwargs)
        self._ACTION_TOKEN_MIN = min(self.action_token_ids)
        self._ACTION_TOKEN_MAX = max(self.action_token_ids)

        lora_cfg = cfg.get("lora", {})
        if bool(lora_cfg.get("enabled", True)):
            model = get_peft_model(
                model,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=int(lora_cfg.get("r", 64)),
                    lora_alpha=int(lora_cfg.get("alpha", 128)),
                    lora_dropout=float(lora_cfg.get("dropout", 0.1)),
                    target_modules=list(
                        lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"])
                    ),
                    bias="none",
                    trainable_token_indices={"embed_tokens": self.action_token_ids},
                ),
            )

        if bool(cfg.get("gradient_checkpointing", True)):
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        self.model = model

        self.vq_model = build_vision_tokenizer("ibq", cfg.vision_tokenizer, device=cfg.get("vq_device", "cpu"))
        self.vq_model.requires_grad_(False).eval()

    def _set_special_tokens(self) -> None:
        token_values = {
            "bos_token": "<|extra_203|>",
            "eos_token": "<|extra_204|>",
            "pad_token": "<|endoftext|>",
            "eol_token": "<|extra_200|>",
            "eof_token": "<|extra_201|>",
            "tms_token": "<|extra_202|>",
            "img_token": "<|image token|>",
            "boi_token": "<|image start|>",
            "eoi_token": "<|image end|>",
            "bss_token": "<|extra_100|>",
            "ess_token": "<|extra_101|>",
        }
        for name, value in token_values.items():
            setattr(self.tokenizer, name, value)

    @property
    def input_device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @torch.no_grad()
    def encode_image_codes(self, image: Image.Image | np.ndarray) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype(np.uint8))
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        parameter = next(self.vq_model.parameters())
        pixels = torch.from_numpy(np.asarray(image).copy()).to(parameter.device, parameter.dtype)
        pixels = (pixels / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
        _, _, info = self.vq_model.encode(pixels)
        return info[-1].reshape(self.image_size // 16, self.image_size // 16)

    def format_image_codes(self, codes) -> list[int]:
        codes = torch.as_tensor(codes)
        if (codes.shape != (self.image_size // 16, self.image_size // 16)
                or codes.dtype.is_floating_point or codes.dtype == torch.bool
                or (codes < 0).any() or (codes >= 131072).any()):
            raise ValueError("Invalid IBQ codebook grid")
        return self.tokenizer.encode(_format_image_tokens(self.tokenizer, codes), add_special_tokens=False)

    @torch.no_grad()
    def encode_image(self, image: Image.Image | np.ndarray) -> list[int]:
        return self.format_image_codes(self.encode_image_codes(image))

    def build_prompt_ids(self, images: Iterable[Image.Image | np.ndarray], instruction: str, image_tokens=None) -> list[int]:
        prefix = (
            "You are a helpful assistant for vla task. USER: Given the current camera views, "
            f"how to perform the following task? {instruction}"
        )
        ids = [self.tokenizer.bos_token_id]
        ids += self.tokenizer.encode(prefix, add_special_tokens=False)
        for tokens in (image_tokens if image_tokens is not None else (self.encode_image(image) for image in images)):
            ids += list(tokens)
        ids += self.tokenizer.encode(" ASSISTANT: ", add_special_tokens=False)
        ids += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bss_token)]
        if any(self._ACTION_TOKEN_MIN <= token <= self._ACTION_TOKEN_MAX for token in ids):
            raise ValueError("Prompt contains a text token reserved for FAST actions; choose a non-conflicting mapping")
        return ids

    def build_training_inputs(self, images, instructions, fast_tokens, image_tokens=None):
        ess_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.ess_token)
        rows, labels = [], []
        if image_tokens is not None and len(image_tokens) != len(images):
            raise ValueError("Cached image batch size mismatch")
        for index, (sample_images, instruction, token_ids) in enumerate(zip(images, instructions, fast_tokens)):
            prompt = self.build_prompt_ids(sample_images, instruction, None if image_tokens is None else image_tokens[index])
            if not token_ids or any(not 0 <= int(idx) < ACTION_TOKEN_COUNT for idx in token_ids):
                raise ValueError(f"FAST produced invalid action tokens: {token_ids}")
            answer = [self.action_token_ids[int(idx)] for idx in token_ids] + [ess_id, self.tokenizer.eos_token_id]
            row = prompt + answer
            if len(row) > self.max_length:
                raise ValueError(f"Emu3.5 sequence length {len(row)} exceeds max_length={self.max_length}")
            rows.append(row)
            labels.append([-100] * len(prompt) + answer)

        width = max(map(len, rows))
        input_ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        label_ids = torch.full_like(input_ids, -100)
        for index, (row, target) in enumerate(zip(rows, labels)):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
            label_ids[index, : len(target)] = torch.tensor(target)
        return {
            "input_ids": input_ids.to(self.input_device),
            "attention_mask": attention_mask.to(self.input_device),
            "labels": label_ids.to(self.input_device),
        }

    @torch.inference_mode()
    def generate_fast_tokens(self, images, instructions) -> list[list[int]]:
        ess_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.ess_token)
        allowed = self.action_token_ids + [ess_id]
        results = []
        for sample_images, instruction in zip(images, instructions):
            prompt = self.build_prompt_ids(sample_images, instruction)
            input_ids = torch.tensor([prompt], dtype=torch.long, device=self.input_device)
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=ess_id,
                pad_token_id=self.tokenizer.pad_token_id,
                prefix_allowed_tokens_fn=lambda _batch, _ids: allowed,
            )
            continuation = output[0, len(prompt) :].tolist()
            action_ids = [token for token in continuation if token != ess_id]
            if not action_ids or any(token not in self.action_token_ids for token in action_ids):
                raise RuntimeError(f"Emu3.5 generated an invalid FAST sequence: {continuation}")
            results.append([token - self._ACTION_TOKEN_MIN for token in action_ids])
        return results
