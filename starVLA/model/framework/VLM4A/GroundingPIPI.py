"""Layer-wise flow-matching actions conditioned on GroundingPI."""

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
from starVLA.model.framework.VLM4A._groundingpi_config import grounding_config
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("GroundingPIPI")
class GroundingPIPI(Qwen_PI):
    def __init__(self, config=None, **kwargs):
        super().__init__(grounding_config(config, "GroundingPIPI"), **kwargs)

    def _encode_vl_hidden_states(self, batch_images, instructions):
        states, mask = super()._encode_vl_hidden_states(batch_images, instructions)
        dtype = next(self.action_model.parameters()).dtype
        return [state.to(dtype=dtype) for state in states], mask
