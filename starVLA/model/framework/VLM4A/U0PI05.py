"""U0 backbone with a pi0.5-style continuous flow-matching action head.

The first version intentionally omits FAST CE supervision.  U0 provides the
multimodal prefix, while the layer-wise DiT predicts a continuous action chunk.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    get_action_model,
)
from starVLA.model.modules.vlm.U0 import U0Interface
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class U0PI05DefaultConfig:
    name: str = "U0PI05"
    u0: dict = field(default_factory=lambda: {
        "base_vlm": "/root/nas/code/Xiaomi-Robotics-U0/training/models/Xiaomi-Robotics-U0-4B",
        "vision_tokenizer": "playground/Pretrained_models/Emu3.5-VisionTokenizer",
        "image_size": 224,
        "max_length": 2048,
        "max_new_tokens": 64,
        "action_token_start_id": 149595,
        "loss_chunk_tokens": 64,
        "attn_implementation": "flash_attention_2",
        "gradient_checkpointing": True,
        "use_cached_vision": True,
    })
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "LayerwiseFM",
        "action_dim": 7,
        "action_horizon": 8,
        "state_dim": 8,
        "num_inference_timesteps": 4,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "num_target_vision_tokens": 32,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "diffusion_model_cfg": {
            "action_dit_hidden_dim": 1024,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": False,
            "norm_type": "ada_norm",
            "positional_embeddings": None,
            "attention_head_dim": 64,
        },
    })


@FRAMEWORK_REGISTRY.register("U0PI05")
class U0PI05(baseframework):
    """U0 multimodal prefix plus layer-wise continuous action flow matching."""

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = merge_framework_config(U0PI05DefaultConfig, config)
        self.u0_interface = U0Interface(self.config)

        decoder_config = self.u0_interface.model.get_decoder().config
        self.u0_hidden_dim = int(decoder_config.hidden_size)
        self.num_u0_layers = int(decoder_config.num_hidden_layers)
        dit_hidden_dim = int(
            self.config.framework.action_model.diffusion_model_cfg.get("action_dit_hidden_dim", self.u0_hidden_dim)
        )
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=dit_hidden_dim,
            num_dit_layers=self.num_u0_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)
        if self.num_action_dit_layers != self.num_u0_layers:
            raise RuntimeError(
                f"U0/DiT layer mismatch: U0 has {self.num_u0_layers}, "
                f"action DiT has {self.num_action_dit_layers}"
            )
        self.project_layers = nn.ModuleList([
            nn.Identity()
            if self.u0_hidden_dim == dit_hidden_dim
            else nn.Sequential(
                nn.LayerNorm(self.u0_hidden_dim),
                nn.Linear(self.u0_hidden_dim, dit_hidden_dim),
            )
            for _ in range(self.num_action_dit_layers)
        ])
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.action_dim = int(self.config.framework.action_model.action_dim)

    @staticmethod
    def _state_instruction(instruction, state):
        if state is None:
            return instruction
        values = np.asarray(state, dtype=np.float32)
        if values.ndim == 1:
            values = values[None]
        if values.ndim != 2 or values.shape[0] < 1 or not np.isfinite(values[-1]).all():
            raise ValueError("U0PI05 state must be a finite vector or [T, state_dim] array")
        bins = np.digitize(values[-1], bins=np.linspace(-1, 1, 257)[:-1]) - 1
        return f"{instruction} [STATE] {' '.join(map(str, bins.tolist()))} [ACTION]"

    def _prefix_batch(self, examples):
        if not isinstance(examples, list) or not examples:
            raise ValueError("U0PI05 expects a nonempty list of examples")
        if any(not isinstance(example.get("lang"), str) for example in examples):
            raise ValueError("Each U0PI05 example requires a string instruction")

        cached = [example.get("image_codes") for example in examples]
        use_cache = all(codes is not None for codes in cached)
        if any((codes is not None) != use_cache for codes in cached):
            raise ValueError("U0PI05 requires image_codes for every example or none")
        if use_cache:
            images = [[] for _ in examples]
            image_tokens = [
                [self.u0_interface.format_image_codes(view) for view in codes]
                for codes in cached
            ]
        else:
            images = [to_pil_preserve(example["image"]) for example in examples]
            image_tokens = None
        instructions = [
            self._state_instruction(example["lang"], example.get("state"))
            for example in examples
        ]
        inputs = self.u0_interface.build_prefix_inputs(images, instructions, image_tokens=image_tokens)
        outputs = self.u0_interface.encode_prefix_hidden(inputs)
        if outputs.hidden_states is None:
            raise RuntimeError("U0 did not return hidden states")
        hidden_states = list(outputs.hidden_states[-self.num_action_dit_layers:])
        projected = [
            projector(hidden.to(dtype=next(projector.parameters()).dtype)
                      if not isinstance(projector, nn.Identity) else hidden)
            for projector, hidden in zip(self.project_layers, hidden_states, strict=True)
        ]
        return projected, inputs["attention_mask"].to(dtype=torch.bool)

    def forward(self, examples=None, **kwargs):
        hidden_states, attention_mask = self._prefix_batch(examples)
        actions = []
        for example in examples:
            value = np.asarray(example["action"], dtype=np.float32)
            if value.ndim != 2 or value.shape[1] != self.action_dim or value.shape[0] < self.action_horizon:
                raise ValueError(
                    f"Expected action shape [T, {self.action_dim}] with T >= {self.action_horizon}, got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError("U0PI05 actions must be finite")
            actions.append(value[-self.action_horizon:])
        action_tensor = torch.as_tensor(
            np.asarray(actions), device=hidden_states[-1].device, dtype=hidden_states[-1].dtype
        )
        loss = self.action_model(
            hidden_states,
            action_tensor,
            state=None,
            encoder_attention_mask=attention_mask,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"U0PI05 produced a nonfinite flow loss: {loss}")
        return {"action_loss": loss}

    @torch.inference_mode()
    def predict_action(self, examples=None, **kwargs):
        if isinstance(examples, dict):
            examples = [examples]
        hidden_states, attention_mask = self._prefix_batch(examples)
        actions = self.action_model.predict_action(
            hidden_states,
            state=None,
            encoder_attention_mask=attention_mask,
        )
        result = actions.detach().cpu().numpy()
        if result.shape != (len(examples), self.action_horizon, self.action_dim) or not np.isfinite(result).all():
            raise RuntimeError(f"U0PI05 decoded invalid actions: shape={result.shape}")
        return {"normalized_actions": result}
