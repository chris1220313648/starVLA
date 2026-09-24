"""Full-parameter U0 fine-tuning for FAST action prediction."""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model
from starVLA.model.modules.vlm.U0 import U0Interface, fast_token_constraint
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class U0FastDefaultConfig:
    name: str = "U0Fast"
    u0: dict = field(default_factory=lambda: {
        "base_vlm": "/root/nas/code/Xiaomi-Robotics-U0/training/models/Xiaomi-Robotics-U0-4B",
        "vision_tokenizer": "playground/Pretrained_models/Emu3.5-VisionTokenizer",
        "image_size": 224, "max_length": 2048, "max_new_tokens": 64,
        "action_token_start_id": 149595, "loss_chunk_tokens": 64,
        "attn_implementation": "flash_attention_2", "gradient_checkpointing": True,
    })
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "FAST", "action_dim": 7, "action_horizon": 8,
        "fast_tokenizer_name": "playground/Pretrained_models/fast",
    })


@FRAMEWORK_REGISTRY.register("U0Fast")
class U0Fast(baseframework):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = merge_framework_config(U0FastDefaultConfig, config)
        self.u0_interface = U0Interface(self.config)
        self.register_buffer("action_token_ids", torch.tensor(self.u0_interface.action_token_ids))
        self.action_model = get_action_model(self.config)
        cfg = self.config.framework.action_model
        self.action_horizon, self.action_dim = int(cfg.action_horizon), int(cfg.action_dim)
        self.action_model.fast_tokenizer.time_horizon = self.action_horizon
        self.action_model.fast_tokenizer.action_dim = self.action_dim
        self.sequence_packer = None
        if int(self.config.framework.u0.get('sequence_h', 1)) > 1:
            from starVLA.model.modules.vlm.u0_sequence import SequencePacker, contract_hash
            contract = OmegaConf.to_container(self.config.framework.u0.sequence_contract, resolve=True)
            if contract_hash(contract) != self.config.framework.u0.sequence_contract_sha256:
                raise ValueError('U0 sequence contract hash mismatch')
            if contract['sequence_h'] != int(self.config.framework.u0.sequence_h):
                raise ValueError('sequence_h differs from sequence contract')
            for filename, digest in contract['encoder_sha256'].items():
                with Path(filename).open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                        raise ValueError(f'Encoder identity mismatch: {filename}')
            self.sequence_packer = SequencePacker(self.u0_interface, contract)

    def _observations(self, examples):
        if not isinstance(examples, list) or not examples:
            raise ValueError("U0Fast expects a nonempty list of examples")
        images = [to_pil_preserve(e["image"]) for e in examples]
        if any(not views for views in images) or any(not isinstance(e["lang"], str) for e in examples):
            raise ValueError("Each U0Fast example requires camera views and a string instruction")
        return images, [e["lang"] for e in examples]

    def forward(self, examples=None, **kwargs):
        if self.sequence_packer is not None:
            h = self.sequence_packer.h
            actions = [np.asarray(e['sequence_action']) for e in examples]
            if any(a.shape != (h, 8, 7) or not np.isfinite(a).all() for a in actions):
                raise ValueError('Expected h complete FAST action windows')
            flat = self.action_model.encoder_action2fastoken([a for rows in actions for a in rows])
            tokens = [flat[i*h:(i+1)*h] for i in range(len(examples))]
            inputs = self.sequence_packer.training_inputs(examples, tokens)
            result = self.u0_interface.action_loss(inputs, self.sequence_packer.contract['future_image_weight'])
            result['sequence_tokens'] = inputs['attention_mask'].sum(-1).max().detach()
            if not torch.isfinite(result['action_loss']):
                raise RuntimeError('Nonfinite U0 interleaved loss')
            return result
        cached = bool(self.config.framework.u0.get("use_cached_vision", False))
        if cached:
            if not examples or any("image_codes" not in e or not isinstance(e["lang"], str) for e in examples):
                raise ValueError("U0 cached training requires image_codes and language for every sample")
            images, instructions = [[] for e in examples], [e["lang"] for e in examples]
        else:
            images, instructions = self._observations(examples)
        actions = [np.asarray(e["action"]) for e in examples]
        if any(a.shape != (self.action_horizon, self.action_dim) or not np.isfinite(a).all() for a in actions):
            raise ValueError("U0Fast action shape or values do not match the configured horizon/dimension")
        tokens = self.action_model.encoder_action2fastoken(actions)
        inputs = self.u0_interface.build_training_inputs(
            images, instructions, tokens,
            image_tokens=[[self.u0_interface.format_image_codes(view) for view in e["image_codes"]]
                          for e in examples] if cached else None,
        )
        loss = self.u0_interface.action_loss(inputs)
        if not torch.isfinite(loss):
            raise RuntimeError(f"U0Fast produced a nonfinite action loss: {loss}")
        return {"action_loss": loss}

    @torch.inference_mode()
    def predict_action(self, examples=None, **kwargs):
        if isinstance(examples, dict):
            examples = [examples]
        prompts = None
        if self.sequence_packer is None and isinstance(examples, list) and any("image_codes" in e for e in examples):
            if any("image_codes" not in e or len(e["image_codes"]) == 0 or not isinstance(e.get("lang"), str) for e in examples):
                raise ValueError("U0 cached prediction requires image_codes and language for every sample")
            images, instructions = [[] for e in examples], [e["lang"] for e in examples]
            prompts = [self.u0_interface.build_prompt_ids(
                [], e["lang"], image_tokens=[self.u0_interface.format_image_codes(view) for view in e["image_codes"]],
            ) for e in examples]
        else:
            images, instructions = self._observations(examples)
        if self.sequence_packer is not None:
            prompts = []
            for e, views in zip(examples, images):
                history = []
                for entry in e.get('history', []):
                    if len(self.action_model.fast_tokenizer.bpe_tokenizer.decode(entry['fast_tokens'])) != self.action_horizon * self.action_dim:
                        raise ValueError('History FAST tokens do not encode 56 coefficients')
                    grids = [self.u0_interface.encode_image_codes(v).cpu().numpy() for v in to_pil_preserve(entry['image'])]
                    history.append(dict(state=entry['state'], image_codes=grids, fast_tokens=entry['fast_tokens']))
                grids = [self.u0_interface.encode_image_codes(v).cpu().numpy() for v in views]
                prompts.append(self.sequence_packer.prompt(e['lang'], e['state'], grids, history))
        expected_coefficients = self.action_horizon * self.action_dim
        constraint = fast_token_constraint(
            self.action_model.fast_tokenizer.bpe_tokenizer,
            self.u0_interface.action_token_ids,
            self.u0_interface.tokenizer.convert_tokens_to_ids(self.u0_interface.tokenizer.ess_token),
            expected_coefficients,
        )
        tokens = self.u0_interface.generate_fast_tokens(images, instructions, action_constraint=constraint, prompt_ids=prompts)
        # The upstream FAST processor silently returns zero actions on malformed
        # coefficient lengths. Reject those sequences before calling decode.
        if any(len(self.action_model.fast_tokenizer.bpe_tokenizer.decode(row)) != expected_coefficients for row in tokens):
            raise RuntimeError("U0 FAST sequence does not encode the configured number of action coefficients")
        actions = np.asarray(self.action_model.fast_tokenizer.decode(tokens), dtype=np.float32)
        if actions.shape != (len(examples), self.action_horizon, self.action_dim) or not np.isfinite(actions).all():
            raise RuntimeError(f"U0Fast decoded invalid actions: shape={actions.shape}")
        result = {"normalized_actions": actions}
        if self.sequence_packer is not None:
            result['fast_tokens'] = tokens
        return result

    def checkpoint_state_dict(self, full_state_dict):
        state = {k: v for k, v in full_state_dict.items() if k.startswith("u0_interface.model.")}
        expected = {"u0_interface.model." + k for k in self.u0_interface.model.state_dict()}
        if set(state) != expected:
            raise RuntimeError("Incomplete U0 full-parameter checkpoint")
        state["action_token_ids"] = self.action_token_ids.detach().cpu().clone()
        return state

    def load_checkpoint_state_dict(self, state_dict):
        ids = state_dict.get("action_token_ids")
        if ids is None or not torch.equal(ids.cpu(), self.action_token_ids.cpu()):
            raise RuntimeError("U0 checkpoint action-token mapping is missing or incompatible")
        prefix = "u0_interface.model."
        if any(k != "action_token_ids" and not k.startswith(prefix) for k in state_dict):
            raise RuntimeError("Unexpected U0 checkpoint tensors")
        self.u0_interface.model.load_state_dict(
            {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}, strict=True,
        )
