# Copyright 2026 ByteDance and/or its affiliates.
# Licensed under the Apache License, Version 2.0.

"""Minimal Emu3.5 runtime vendored from UniVR.

The LDM/image-generation modules are intentionally excluded: StarVLA only
needs the causal language model and IBQ image encoder for VLA.
"""

from .configuration_emu3 import Emu3Config
from .modeling_emu3 import Emu3ForCausalLM

__all__ = ["Emu3Config", "Emu3ForCausalLM"]
