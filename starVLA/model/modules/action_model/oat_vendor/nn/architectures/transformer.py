# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Transformer stacks used by action-tokenizer models."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn

from starVLA.model.modules.action_model.oat_vendor.nn.blocks.attention import CrossAttention, SelfAttention
from starVLA.model.modules.action_model.oat_vendor.nn.blocks.transformer_block import (
    Block,
    BlockAdaLN,
    DecoderBlock,
    DecoderBlockAdaLN,
)
from starVLA.model.modules.action_model.oat_vendor.nn.layers.norm import Fp32RMSNorm

__all__ = ["Transformer", "TransformerDecoder"]

_DEFAULT_NORM_LAYER = partial(Fp32RMSNorm, elementwise_affine=False)


def _zero_query_projection(module: nn.Module) -> None:
    if isinstance(module, SelfAttention):
        query_dim = module.qkv.out_features // 3
        nn.init.zeros_(module.qkv.weight[:query_dim])
        if module.qkv.bias is not None:
            nn.init.zeros_(module.qkv.bias[:query_dim])
    elif isinstance(module, CrossAttention):
        nn.init.zeros_(module.q_proj.weight)
        if module.q_proj.bias is not None:
            nn.init.zeros_(module.q_proj.bias)


def _init_transformer_module(
    module: nn.Module,
    *,
    name: str,
    weight_init_style: str,
) -> None:
    if isinstance(module, nn.Linear):
        if "adaLN_modulation" in name:
            nn.init.zeros_(module.weight)
        elif weight_init_style == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif weight_init_style == "trunc_normal":
            nn.init.trunc_normal_(module.weight, std=0.02)
        else:
            raise ValueError(f"Unsupported weight init: {weight_init_style}")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Transformer(nn.Module):
    """Stack of encoder-style transformer blocks."""

    def __init__(
        self,
        dim: int = 768,
        depth: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        mlp_bias: bool = False,
        drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: type[nn.Module] = nn.SiLU,
        norm_layer: Callable[[int], nn.Module] = _DEFAULT_NORM_LAYER,
        gated_mlp: bool = True,
        qk_norm: bool = True,
        weight_init_style: str = "xavier",
        zero_init_query_proj: bool = False,
        use_adaLN: bool = False,
        adaLN_expansion: int = 1,
    ) -> None:
        super().__init__()
        block_cls = BlockAdaLN if use_adaLN else Block
        drop_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    dim=dim,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_bias=mlp_bias,
                    drop=drop,
                    drop_path=float(drop_rates[idx]),
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    gated_mlp=gated_mlp,
                    qk_norm=qk_norm,
                    adaLN_expansion=adaLN_expansion,
                )
                for idx in range(depth)
            ]
        )
        self.weight_init_style = weight_init_style
        self.zero_init_query_proj = bool(zero_init_query_proj)
        self.init_weights_sp()

    def init_weights_sp(self) -> None:
        """Initialize linear and LayerNorm modules using the configured scheme."""
        for name, module in self.named_modules():
            _init_transformer_module(
                module, name=name, weight_init_style=self.weight_init_style
            )
        if self.zero_init_query_proj:
            for module in self.modules():
                _zero_query_projection(module)

    def forward(
        self,
        x: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        adaLN_emb: torch.Tensor | None = None,
        adaLN_packing_fn: Callable | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x,
                block_mask=block_mask,
                adaLN_emb=adaLN_emb,
                adaLN_packing_fn=adaLN_packing_fn,
            )
        return x


class TransformerDecoder(nn.Module):
    """Stack of decoder blocks with cross-attention to a context sequence."""

    def __init__(
        self,
        dim: int = 768,
        depth: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        mlp_bias: bool = False,
        drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: type[nn.Module] = nn.SiLU,
        norm_layer: Callable[[int], nn.Module] = _DEFAULT_NORM_LAYER,
        gated_mlp: bool = True,
        qk_norm: bool = True,
        weight_init_style: str = "xavier",
        zero_init_query_proj: bool = False,
        use_adaLN: bool = False,
        adaLN_expansion: int = 1,
        enable_self_attn: bool = True,
    ) -> None:
        super().__init__()
        block_cls = DecoderBlockAdaLN if use_adaLN else DecoderBlock
        drop_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    dim=dim,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_bias=mlp_bias,
                    drop=drop,
                    drop_path=float(drop_rates[idx]),
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    gated_mlp=gated_mlp,
                    qk_norm=qk_norm,
                    enable_self_attn=enable_self_attn,
                    adaLN_expansion=adaLN_expansion,
                )
                for idx in range(depth)
            ]
        )
        self.weight_init_style = weight_init_style
        self.zero_init_query_proj = bool(zero_init_query_proj)
        self.init_weights_sp()

    def init_weights_sp(self) -> None:
        """Initialize linear and LayerNorm modules using the configured scheme."""
        for name, module in self.named_modules():
            _init_transformer_module(
                module, name=name, weight_init_style=self.weight_init_style
            )
        if self.zero_init_query_proj:
            for module in self.modules():
                _zero_query_projection(module)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        context_block_mask: torch.Tensor | None = None,
        adaLN_emb: torch.Tensor | None = None,
        adaLN_packing_fn: Callable | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x,
                context=context,
                block_mask=block_mask,
                context_block_mask=context_block_mask,
                adaLN_emb=adaLN_emb,
                adaLN_packing_fn=adaLN_packing_fn,
            )
        return x
