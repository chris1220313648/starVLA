"""Autoregressive FAST action-token prediction with GroundingPI."""

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenFast import Qwenvl_Fast
from starVLA.model.framework.VLM4A._groundingpi_config import grounding_config
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("GroundingPIFast")
class GroundingPIFast(Qwenvl_Fast):
    def __init__(self, config=None, **kwargs):
        super().__init__(grounding_config(config, "GroundingPIFast"), **kwargs)
        if self.action_model.fast_tokenizer.vocab_size != self.config.framework.qwenvl.action_vocab_size:
            raise ValueError("FAST tokenizer vocabulary size must match qwenvl.action_vocab_size")

    def forward(self, examples=None, **kwargs):
        # Keep invalid losses visible rather than the legacy FAST zero-loss fallback.
        examples = [dict(e, action=np.asarray(e["action"])[-self.action_horizon:]) for e in examples]
        action_ids = self.action_model.encoder_action2fastoken([e["action"] for e in examples])
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=[e["image"] for e in examples], instructions=[e["lang"] for e in examples],
            solutions=[self.map_fast_token_to_vlm_action(ids) for ids in action_ids],
        )
        output = self.qwen_vl_interface(**inputs, return_dict=True, use_cache=False)
        if output.loss is None or not torch.isfinite(output.loss):
            raise FloatingPointError("GroundingPI FAST produced an invalid action loss")
        return {"action_loss": output.loss}

    @torch.inference_mode()
    def predict_action(self, examples=None, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=[to_pil_preserve(e["image"]) for e in examples],
            instructions=[e["lang"] for e in examples],
        )
        # Shared VLA clients also send diffusion-only options.
        kwargs.pop("use_ddim", None)
        kwargs.pop("num_ddim_steps", None)
        do_sample = kwargs.pop("do_sample", False)
        use_cache = kwargs.pop("use_cache", True)
        generated = self.qwen_vl_interface.generate(
            **inputs, max_new_tokens=kwargs.pop("max_new_tokens", 256),
            do_sample=do_sample, use_cache=use_cache, **kwargs,
        )[:, inputs["input_ids"].shape[1]:]
        tokens = self._decode_action_tokens(self._extract_action_token_ids(generated))
        if any(not ids for ids in tokens):
            raise ValueError("GroundingPI generated no action tokens; train the FAST policy before action inference")
        coefficient_count = self.action_horizon * self.config.framework.action_model.action_dim
        for ids in tokens:
            if len(self.action_model.fast_tokenizer.bpe_tokenizer.decode(ids)) != coefficient_count:
                raise ValueError("FAST generated an invalid coefficient count; refusing the tokenizer's zero-action fallback")
        actions = np.asarray(self.action_model.fast_tokenizer.decode(tokens))
        expected = (len(examples), self.action_horizon, self.config.framework.action_model.action_dim)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Invalid FAST actions: expected {expected}, got {actions.shape}")
        return {"normalized_actions": actions}
