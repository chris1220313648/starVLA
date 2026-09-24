"""Emu3.5 backbone with autoregressive FAST action tokens."""

from dataclasses import dataclass, field

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model
from starVLA.model.modules.vlm.Emu3_5 import ACTION_TOKEN_START_ID, Emu3_5Interface
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class EmuFastDefaultConfig:
    name: str = "EmuFast"
    emu: dict = field(
        default_factory=lambda: {
            "base_vlm": "playground/Pretrained_models/Emu3.5",
            "vision_tokenizer": "playground/Pretrained_models/Emu3.5-VisionTokenizer",
            "image_size": 224,
            "action_token_start_id": ACTION_TOKEN_START_ID,
            "max_length": 2048,
            "max_new_tokens": 64,
            "attn_implementation": "flash_attention_2",
            "gradient_checkpointing": True,
            "device_map": None,
            "vq_device": "cpu",
            "lora": {
                "enabled": True,
                "r": 64,
                "alpha": 128,
                "dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
            },
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "FAST",
            "action_dim": 7,
            "action_horizon": 8,
            "fast_tokenizer_name": "playground/Pretrained_models/fast",
        }
    )


@FRAMEWORK_REGISTRY.register("EmuFast")
class EmuFast(baseframework):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = merge_framework_config(EmuFastDefaultConfig, config)
        self.emu_vl_interface = Emu3_5Interface(self.config)
        self.register_buffer("action_token_ids", torch.tensor(self.emu_vl_interface.action_token_ids))
        self.action_model = get_action_model(self.config)
        action_cfg = self.config.framework.action_model
        self.action_horizon = int(action_cfg.action_horizon)
        self.action_dim = int(action_cfg.action_dim)
        self.action_model.fast_tokenizer.time_horizon = self.action_horizon
        self.action_model.fast_tokenizer.action_dim = self.action_dim
        self.manages_own_device = self.config.framework.emu.get("device_map", None) not in (None, "none", "null", "")

    def forward(self, examples=None, **kwargs):
        images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        fast_tokens = self.action_model.encoder_action2fastoken([example["action"] for example in examples])
        inputs = self.emu_vl_interface.build_training_inputs(images, instructions, fast_tokens)
        outputs = self.emu_vl_interface.model(**inputs, return_dict=True)
        if outputs.loss is None or not torch.isfinite(outputs.loss):
            raise RuntimeError(f"EmuFast produced a non-finite action loss: {outputs.loss}")
        return {"action_loss": outputs.loss}

    @torch.inference_mode()
    def predict_action(self, examples=None, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        token_ids = self.emu_vl_interface.generate_fast_tokens(images, instructions)
        actions = np.asarray(self.action_model.fast_tokenizer.decode(token_ids), dtype=np.float32)
        expected = (len(examples), self.action_horizon, self.action_dim)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise RuntimeError(f"FAST decoded actions with shape {actions.shape}; expected {expected}")
        return {"normalized_actions": actions}

    def checkpoint_state_dict(self, full_state_dict):
        adapter = {
            key: value
            for key, value in full_state_dict.items()
            if ".lora_" in key or ".trainable_tokens_" in key
        }
        if not adapter:
            raise RuntimeError("No EmuFast LoRA/trainable-token parameters found while saving")
        adapter["action_token_ids"] = self.action_token_ids.detach().cpu().clone()
        return adapter

    def load_checkpoint_state_dict(self, state_dict):
        saved_ids = state_dict.get("action_token_ids")
        if saved_ids is None or not torch.equal(saved_ids.cpu(), self.action_token_ids.cpu()):
            raise RuntimeError(
                "EmuFast checkpoint action-token mapping is missing or incompatible. "
                "Old expanded-vocabulary checkpoints cannot be used with text-token reuse; retrain."
            )
        expected = {name for name, param in self.named_parameters() if param.requires_grad}
        missing = expected.difference(state_dict)
        if missing:
            raise RuntimeError(f"Missing EmuFast trainable checkpoint tensors: {sorted(missing)}")
        result = self.load_state_dict(state_dict, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected EmuFast adapter keys: {result.unexpected_keys}")
        loaded = set(state_dict).intersection(self.state_dict())
        if loaded != set(state_dict):
            raise RuntimeError("Not every EmuFast adapter tensor was restored")
