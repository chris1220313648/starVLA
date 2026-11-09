"""Parallel continuous action regression with GroundingPI."""

from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
from starVLA.model.framework.VLM4A._groundingpi_config import grounding_config
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("GroundingPIOFT")
class GroundingPIOFT(Qwenvl_OFT):
    def __init__(self, config=None, **kwargs):
        super().__init__(grounding_config(config, "GroundingPIOFT"), **kwargs)

    def _gather_action_token_embeddings(self, *args, **kwargs):
        queries = super()._gather_action_token_embeddings(*args, **kwargs)
        return queries.to(dtype=next(self.action_model.parameters()).dtype)
