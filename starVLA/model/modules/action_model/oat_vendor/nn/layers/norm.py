# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Mixed-precision normalization layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Fp32LayerNorm", "Fp32RMSNorm"]


class Fp32LayerNorm(nn.LayerNorm):
    """LayerNorm that computes statistics in fp32 and returns the input dtype."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            weight,
            bias,
            self.eps,
        )
        return output.to(dtype=input.dtype)


class Fp32RMSNorm(nn.Module):
    """RMSNorm with fp32 statistics and optional affine scale."""

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        x = input.float()
        variance = x.pow(2).mean(dim=dims, keepdim=True)
        output = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            output = output * self.weight.float()
        return output.to(dtype=input.dtype)
