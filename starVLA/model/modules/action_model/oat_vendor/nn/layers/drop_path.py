# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Stochastic-depth layers."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["DropPath", "drop_path"]


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Apply per-sample stochastic depth to a residual branch."""
    if drop_prob == 0.0 or not training:
        return x
    if not 0.0 <= drop_prob < 1.0:
        raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}.")

    keep_prob = 1.0 - float(drop_prob)
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    keep_mask = torch.empty(mask_shape, dtype=x.dtype, device=x.device).bernoulli_(
        keep_prob
    )
    return x * keep_mask.div(keep_prob)


class DropPath(nn.Module):
    """Module wrapper for :func:`drop_path`."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"
