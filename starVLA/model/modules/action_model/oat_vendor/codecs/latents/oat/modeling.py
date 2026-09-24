# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""OAT-family action-latent modeling blocks.

BOAT is a schedule-specialized OAT variant, so its codec reuses these
components and changes only the register schedule plus prefix-boundary policy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from functools import lru_cache, partial
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import trunc_normal_

from starVLA.model.modules.action_model.oat_vendor.codecs.latents.oat.schedule import (
    create_register_schedule_mask,
    is_power_of_two,
    powers_of_two,
)
from starVLA.model.modules.action_model.oat_vendor.nn.architectures.transformer import TransformerDecoder
from starVLA.model.modules.action_model.oat_vendor.nn.embeddings.positional.sinusoid import SinusoidPosEmbed
from starVLA.model.modules.action_model.oat_vendor.nn.layers.norm import Fp32RMSNorm

__all__ = [
    "LinearHead",
    "LinearLayer",
    "MaskedNestedDropout",
    "RegisterQueryEncoder",
    "SampleEmbedder",
    "SinglePassDecoder",
]

_DEFAULT_NORM_LAYER = partial(Fp32RMSNorm, elementwise_affine=False)


def _str_to_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    key = str(value).strip().lower()
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Invalid dtype string representation: {value!r}")
    return mapping[key]


def _get_autocast_context(
    x: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    dtype_override: torch.dtype | None,
) -> Any:
    if isinstance(x, list | tuple):
        if len(x) == 0:
            return nullcontext()
        device_type = x[0].device.type
    else:
        device_type = x.device.type

    if dtype_override is None or dtype_override == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype_override, enabled=True)


class LinearLayer(nn.Module):
    """Projection layer used inside the OAT-family codec."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        weight_init_style: str = "xavier",
        dtype_override: str | None = None,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.dtype_override = _str_to_dtype(dtype_override)
        self.proj = nn.Linear(self.dim_in, self.dim_out, bias=proj_bias)
        self.weight_init_style = weight_init_style
        self.init_weights_sp()

    def init_weights_sp(self) -> None:
        if self.weight_init_style == "zero":
            nn.init.constant_(self.proj.weight, 0)
        elif self.weight_init_style == "xavier":
            nn.init.xavier_uniform_(self.proj.weight)
        elif self.weight_init_style == "trunc_normal":
            nn.init.trunc_normal_(self.proj.weight, std=0.02)
        else:
            raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _get_autocast_context(x, self.dtype_override):
            return self.proj(x)


class LinearHead(nn.Module):
    """Readout head used by the OAT-family codec."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        weight_init_style: str = "zero",
        norm_layer: Callable[[int], nn.Module] | None = _DEFAULT_NORM_LAYER,
        dtype_override: str | None = None,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.dim_in = int(dim)
        self.dim_out = int(dim_out)
        self.dtype_override = _str_to_dtype(dtype_override)
        self.norm = norm_layer(self.dim_in) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(self.dim_in, self.dim_out, bias=proj_bias)
        self.weight_init_style = weight_init_style
        self.init_weights_sp()

    def init_weights_sp(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if self.weight_init_style == "zero" or "adaLN_modulation" in name:
                    nn.init.constant_(module.weight, 0)
                elif self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(module.weight, std=0.02)
                else:
                    raise ValueError(
                        f"Unsupported weight init: {self.weight_init_style}"
                    )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _get_autocast_context(x, self.dtype_override):
            return self.proj(self.norm(x))


class SampleEmbedder(nn.Module):
    """Project action samples into the codec embedding dimension."""

    patch_proj: nn.Linear | nn.Identity

    def __init__(
        self,
        dim_in: int,
        dim_out: int | None = None,
        weight_init_style: str = "xavier",
    ):
        super().__init__()
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out) if dim_out is not None else None
        if self.dim_out is not None:
            self.patch_proj = nn.Linear(self.dim_in, self.dim_out, bias=True)
        else:
            self.dim_out = self.dim_in
            self.patch_proj = nn.Identity()
        self.weight_init_style = weight_init_style
        self.init_weights_sp()

    def init_weights_sp(self) -> None:
        patch_proj = self.patch_proj
        if not isinstance(patch_proj, nn.Linear):
            return
        if self.weight_init_style == "zero":
            nn.init.constant_(patch_proj.weight, 0)
        elif self.weight_init_style == "xavier":
            nn.init.xavier_uniform_(patch_proj.weight)
        elif self.weight_init_style == "trunc_normal":
            nn.init.trunc_normal_(patch_proj.weight, std=0.02)
        else:
            raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
        if patch_proj.bias is not None:
            nn.init.constant_(patch_proj.bias, 0)

    @torch.compiler.disable
    def forward(self, x: Tensor) -> Tensor:
        return self.patch_proj(x)


@lru_cache(maxsize=128)
def _compute_power_biased_weights(
    sample_horizon: int, power: float, device: str
) -> torch.Tensor:
    weights = (
        torch.arange(1, sample_horizon + 1, dtype=torch.float32, device=device) ** power
    )
    return weights / weights.sum()


class MaskedNestedDropout(nn.Module):
    """Nested latent dropout for prefix-trained OAT-family decoders."""

    def __init__(
        self,
        dim: int,
        size_sampling_mode: str = "uniform",
    ):
        super().__init__()
        self.dim = int(dim)
        self.size_sampling_mode = size_sampling_mode

        if self.size_sampling_mode != "disable":
            self.dropout_mask_token = nn.Parameter(
                torch.randn(self.dim), requires_grad=True
            )
            trunc_normal_(self.dropout_mask_token, std=0.02)

    def sample_keep_k(
        self, batch_size: int, sample_horizon: int, device: torch.device
    ) -> torch.Tensor:
        if self.size_sampling_mode == "uniform":
            keep_ks = torch.randint(
                low=1, high=sample_horizon + 1, size=(batch_size,), device=device
            )
        elif self.size_sampling_mode == "pow2":
            if not is_power_of_two(sample_horizon):
                raise ValueError(
                    "pow2 nested dropout requires a power-of-two latent horizon."
                )
            pow2_vals = torch.tensor(powers_of_two(1, sample_horizon), device=device)
            idx = torch.randint(0, len(pow2_vals), (batch_size,), device=device)
            keep_ks = pow2_vals[idx]
        elif self.size_sampling_mode == "uniform_pow2":
            ks = torch.randint(1, sample_horizon + 1, (batch_size,), device=device)
            keep_ks = torch.pow(2, torch.ceil(torch.log2(ks.float()))).to(
                dtype=ks.dtype
            )
        elif self.size_sampling_mode.endswith("_biased"):
            power_map = {
                "linear_biased": 1.0,
                "quadratic_biased": 2.0,
                "cubic_biased": 3.0,
            }
            power = power_map[self.size_sampling_mode]
            weights = _compute_power_biased_weights(sample_horizon, power, str(device))
            indices = torch.multinomial(weights, batch_size, replacement=True)
            keep_ks = indices + 1
        else:
            raise ValueError(
                f"size_sampling_mode {self.size_sampling_mode} not defined."
            )
        return keep_ks

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        decode_keep_k: list[int] | None = None,
    ) -> torch.Tensor:
        if self.size_sampling_mode == "disable":
            return x

        batch_size, horizon, _ = x.shape
        mask_token = self.dropout_mask_token.to(device=x.device, dtype=x.dtype)
        mask: torch.Tensor | None = None
        if decode_keep_k is not None:
            mask = torch.tensor(decode_keep_k, device=x.device).unsqueeze(
                1
            ) <= torch.arange(horizon, device=x.device).unsqueeze(0)
        elif self.training:
            keep_ks = self.sample_keep_k(batch_size, horizon, x.device)
            mask = keep_ks.unsqueeze(1) <= torch.arange(
                horizon, device=x.device
            ).unsqueeze(0)

        if mask is None:
            return x
        return torch.where(mask.unsqueeze(-1), mask_token.view(1, 1, -1), x)


class RegisterQueryEncoder(nn.Module):
    """Schedule-aware register-query encoder for OAT-family codecs."""

    def __init__(
        self,
        sample_dim: int,
        sample_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        latent_dim: int,
        register_schedule: Sequence[int],
    ):
        super().__init__()
        schedule = tuple(int(v) for v in register_schedule)
        if not schedule or any(v <= 0 for v in schedule):
            raise ValueError(f"register_schedule must be positive, got {schedule}.")

        self.sample_emb = SampleEmbedder(sample_dim, emb_dim)
        self.pos_emb = SinusoidPosEmbed(emb_dim, max_shape=[sample_horizon])
        self.registers = nn.Parameter(torch.randn(sum(schedule), emb_dim))
        self.register_pos_emb = SinusoidPosEmbed(emb_dim, max_shape=[sum(schedule)])
        self.transformer = TransformerDecoder(
            dim=emb_dim,
            depth=depth,
            head_dim=head_dim,
            drop=pdropout,
        )
        self.head = LinearHead(emb_dim, latent_dim)
        self.register_schedule = schedule
        self.latent_horizon = sum(schedule)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        batch_size, _, _ = sample.shape
        sample_emb = self.sample_emb(sample)
        sample_emb = sample_emb + self.pos_emb(
            [sample.shape[1]], device=sample_emb.device, dtype=sample_emb.dtype
        )
        registers = self.registers.unsqueeze(0).expand(batch_size, -1, -1)
        registers = registers + self.register_pos_emb(
            [self.latent_horizon], device=registers.device, dtype=registers.dtype
        )
        register_mask = create_register_schedule_mask(
            self.register_schedule, sample.device
        )
        x = self.transformer(
            x=registers,
            context=sample_emb,
            block_mask=register_mask,
            context_block_mask=None,
        )
        return self.head(x)


class SinglePassDecoder(nn.Module):
    """OAT-family decoder with latent-only action cross-attention."""

    def __init__(
        self,
        sample_dim: int,
        sample_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        latent_dropout_mode: str,
        latent_dim: int,
        latent_horizon: int,
    ):
        super().__init__()

        self.sample_pos_emb = SinusoidPosEmbed(emb_dim, max_shape=[sample_horizon])
        self.latent_pos_emb = SinusoidPosEmbed(
            emb_dim,
            max_shape=[latent_horizon],
        )
        self.nested_dropout = MaskedNestedDropout(
            emb_dim, size_sampling_mode=latent_dropout_mode
        )
        self.decoder: nn.Module = TransformerDecoder(
            dim=emb_dim,
            depth=depth,
            head_dim=head_dim,
            drop=pdropout,
            enable_self_attn=False,
        )
        self.latent_proj = LinearLayer(latent_dim, emb_dim)
        self.head = LinearHead(emb_dim, sample_dim)

        self.sample_dim = int(sample_dim)
        self.sample_horizon = int(sample_horizon)
        self.latent_horizon = int(latent_horizon)
        self.emb_dim = int(emb_dim)

    def forward(
        self,
        latents: torch.Tensor,
        decode_keep_k: list[int] | None = None,
    ) -> torch.Tensor:
        latents = self.latent_proj(latents)
        x = self.sample_pos_emb(
            [self.sample_horizon],
            device=latents.device,
            dtype=latents.dtype,
        ).expand(latents.shape[0], -1, -1)

        latents = self.nested_dropout(latents, decode_keep_k=decode_keep_k)
        latents = latents + self.latent_pos_emb(
            [self.latent_horizon], device=latents.device, dtype=latents.dtype
        )

        x = self.decoder(x, latents)
        return self.head(x)
