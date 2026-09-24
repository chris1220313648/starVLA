# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Transformer encoder and decoder blocks."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.action_model.oat_vendor.nn.blocks.attention import CrossAttention, SelfAttention
from starVLA.model.modules.action_model.oat_vendor.nn.blocks.mlp import GatedMlp, Mlp
from starVLA.model.modules.action_model.oat_vendor.nn.layers.drop_path import DropPath
from starVLA.model.modules.action_model.oat_vendor.nn.layers.norm import Fp32RMSNorm

__all__ = ["Block", "BlockAdaLN", "DecoderBlock", "DecoderBlockAdaLN"]

_DEFAULT_NORM_LAYER = partial(Fp32RMSNorm, elementwise_affine=False)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN shift/scale modulation."""
    return x * (1.0 + scale) + shift


def expand_to_padded_seq(emb: torch.Tensor, padded_seq: torch.Tensor) -> torch.Tensor:
    """Right-pad a packed embedding sequence to match a padded sequence length."""
    if emb.shape[1] > padded_seq.shape[1]:
        raise ValueError(
            "packed embedding is longer than padded sequence: "
            f"{emb.shape[1]} > {padded_seq.shape[1]}."
        )
    pad_tokens = padded_seq.shape[1] - emb.shape[1]
    if pad_tokens == 0:
        return emb
    return F.pad(emb, (0, 0, 0, pad_tokens))


def _resolve_num_heads(
    *,
    dim: int,
    num_heads: int | None,
    head_dim: int | None,
) -> int:
    if num_heads is None and head_dim is None:
        raise ValueError("Either num_heads or head_dim must be provided.")
    if num_heads is None:
        if head_dim is None:
            raise ValueError("head_dim must be provided when num_heads is None.")
        if dim % head_dim != 0:
            raise ValueError(f"dim={dim} must be divisible by head_dim={head_dim}.")
        return dim // head_dim
    return int(num_heads)


def _drop_path_module(drop_path: float) -> nn.Module:
    return DropPath(drop_path) if drop_path > 0.0 else nn.Identity()


class Block(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int | None = None,
        head_dim: int | None = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        mlp_bias: bool = False,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = _DEFAULT_NORM_LAYER,
        gated_mlp: bool = False,
        qk_norm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        self.dim = int(dim)
        resolved_heads = _resolve_num_heads(
            dim=self.dim, num_heads=num_heads, head_dim=head_dim
        )

        self.norm1 = norm_layer(self.dim)
        self.attn = SelfAttention(
            self.dim,
            num_heads=resolved_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            proj_drop=drop,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
        )
        self.drop_path = _drop_path_module(drop_path)
        self.norm2 = norm_layer(self.dim)

        mlp_cls = GatedMlp if gated_mlp else Mlp
        self.mlp = mlp_cls(
            in_features=self.dim,
            hidden_features=int(self.dim * mlp_ratio),
            act_layer=act_layer,
            bias=mlp_bias,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        x = x + self.drop_path(self.attn(self.norm1(x), block_mask=block_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BlockAdaLN(Block):
    """Encoder block with adaLN-zero residual gates."""

    def __init__(self, adaLN_expansion: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adaLN_expansion = int(adaLN_expansion)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, self.adaLN_expansion * 6 * self.dim, bias=True),
        )
        modulation = cast(nn.Linear, self.adaLN_modulation[-1])
        nn.init.zeros_(modulation.weight)
        nn.init.zeros_(modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        adaLN_emb: torch.Tensor | None = None,
        adaLN_packing_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if adaLN_emb is None or adaLN_packing_fn is None:
            raise ValueError("BlockAdaLN requires adaLN_emb and adaLN_packing_fn.")

        packed = adaLN_packing_fn(self.adaLN_modulation(adaLN_emb))
        gate_msa, gate_mlp, shift_msa, scale_msa, shift_mlp, scale_mlp = (
            expand_to_padded_seq(packed, x).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.drop_path(
            self.attn(modulate(self.norm1(x), shift_msa, scale_msa), block_mask)
        )
        x = x + gate_mlp * self.drop_path(
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        )
        return x


class DecoderBlock(nn.Module):
    """Pre-norm decoder block with optional self-attention and cross-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int | None = None,
        head_dim: int | None = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        mlp_bias: bool = False,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = _DEFAULT_NORM_LAYER,
        gated_mlp: bool = False,
        qk_norm: bool = False,
        enable_self_attn: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        self.dim = int(dim)
        self.enable_self_attn = bool(enable_self_attn)
        resolved_heads = _resolve_num_heads(
            dim=self.dim, num_heads=num_heads, head_dim=head_dim
        )

        self.norm1 = norm_layer(self.dim)
        self.self_attn: SelfAttention | None = None
        if self.enable_self_attn:
            self.self_attn = SelfAttention(
                self.dim,
                num_heads=resolved_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                proj_drop=drop,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
            )
        self.cross_attn = CrossAttention(
            self.dim,
            num_heads=resolved_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            proj_drop=drop,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
        )
        self.query_norm = norm_layer(self.dim)
        self.context_norm = norm_layer(self.dim)
        self.drop_path = _drop_path_module(drop_path)
        self.norm2 = norm_layer(self.dim)

        mlp_cls = GatedMlp if gated_mlp else Mlp
        self.mlp = mlp_cls(
            in_features=self.dim,
            hidden_features=int(self.dim * mlp_ratio),
            act_layer=act_layer,
            bias=mlp_bias,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        context_block_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if self.self_attn is not None:
            x = x + self.drop_path(self.self_attn(self.norm1(x), block_mask=block_mask))
        x = x + self.drop_path(
            self.cross_attn(
                self.query_norm(x),
                self.context_norm(context),
                block_mask=context_block_mask,
            )
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderBlockAdaLN(DecoderBlock):
    """Decoder block with adaLN-zero residual gates."""

    def __init__(self, adaLN_expansion: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adaLN_expansion = int(adaLN_expansion)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, self.adaLN_expansion * 11 * self.dim, bias=True),
        )
        modulation = cast(nn.Linear, self.adaLN_modulation[-1])
        nn.init.zeros_(modulation.weight)
        nn.init.zeros_(modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        context_block_mask: torch.Tensor | None = None,
        adaLN_emb: torch.Tensor | None = None,
        adaLN_packing_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if adaLN_emb is None or adaLN_packing_fn is None:
            raise ValueError(
                "DecoderBlockAdaLN requires adaLN_emb and adaLN_packing_fn."
            )

        packed = expand_to_padded_seq(
            adaLN_packing_fn(self.adaLN_modulation(adaLN_emb)), x
        )
        gate_msa, gate_mxa, gate_mlp, *mods = packed.chunk(11, dim=-1)
        (
            shift_msa,
            scale_msa,
            shift_mxa_q,
            scale_mxa_q,
            shift_mxa_c,
            scale_mxa_c,
            shift_mlp,
            scale_mlp,
        ) = mods

        if self.self_attn is not None:
            x = x + gate_msa * self.drop_path(
                self.self_attn(
                    modulate(self.norm1(x), shift_msa, scale_msa),
                    block_mask=block_mask,
                )
            )
        x = x + gate_mxa * self.drop_path(
            self.cross_attn(
                modulate(self.query_norm(x), shift_mxa_q, scale_mxa_q),
                modulate(self.context_norm(context), shift_mxa_c, scale_mxa_c),
                block_mask=context_block_mask,
            )
        )
        x = x + gate_mlp * self.drop_path(
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        )
        return x
