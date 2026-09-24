# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Attention blocks used by Praxis transformer modules."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.action_model.oat_vendor.nn.layers.norm import Fp32RMSNorm

__all__ = ["Attention", "CrossAttention", "SelfAttention"]

_DEFAULT_NORM_LAYER = partial(Fp32RMSNorm, elementwise_affine=False)


class Attention(nn.Module):
    """Shared scaled-dot-product attention core."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        proj_bias: bool = False,
        proj_drop: float = 0.0,
        qk_norm: bool = True,
        norm_layer: Callable[[int], nn.Module] = _DEFAULT_NORM_LAYER,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_norm(self._split_heads(q))
        k = self.k_norm(self._split_heads(k))
        v = self._split_heads(v)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)
        return self.proj_drop(self.proj(self._merge_heads(attended)))


class SelfAttention(Attention):
    """Self-attention with a fused QKV projection."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        proj_drop: float = 0.0,
        qk_norm: bool = True,
        norm_layer: Callable[[int], nn.Module] = _DEFAULT_NORM_LAYER,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            proj_bias=proj_bias,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(
        self,
        x: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self._attend(q, k, v, block_mask=block_mask)


class CrossAttention(Attention):
    """Cross-attention with separate query and fused key/value projections."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        proj_drop: float = 0.0,
        qk_norm: bool = True,
        norm_layer: Callable[[int], nn.Module] = _DEFAULT_NORM_LAYER,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            proj_bias=proj_bias,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
        )
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        q = self.q_proj(x)
        k, v = self.kv_proj(context).chunk(2, dim=-1)
        return self._attend(q, k, v, block_mask=block_mask)
