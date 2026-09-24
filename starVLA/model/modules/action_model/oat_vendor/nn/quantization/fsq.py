# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Finite scalar quantization."""

from __future__ import annotations

import random
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["FiniteScalarQuantization"]


def _validate_probability(name: str, value: float) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {probability}.")
    return probability


def round_ste_quant_dropout(z: Tensor, drop_quant_p: float) -> Tensor:
    """Round with a straight-through estimator and optional sample dropout."""
    rounded = z.round()
    if drop_quant_p == 0.0:
        return z + (rounded - z).detach()
    if drop_quant_p == 1.0:
        return z

    batch_size = z.shape[0]
    mask_shape = (batch_size,) + (1,) * (z.ndim - 1)
    skip_mask = torch.empty(mask_shape, dtype=z.dtype, device=z.device).bernoulli_(
        drop_quant_p
    )
    quantized = z + (rounded - z).detach()
    return torch.where(skip_mask.bool(), z, quantized)


class FiniteScalarQuantization(nn.Module):
    """Quantize each latent channel onto a finite scalar grid."""

    def __init__(
        self,
        levels: list[int],
        drop_quant_p: float = 0.0,
        corrupt_tokens_p: float = 0.0,
        min_corrupt_tokens_p: float | None = None,
        apply_corrupt_tokens_p: float = 0.2,
    ) -> None:
        super().__init__()
        if not levels:
            raise ValueError("levels must be non-empty.")
        if any(level < 2 for level in levels):
            raise ValueError(f"all levels must be >= 2, got levels={levels!r}.")

        self.drop_quant_p = _validate_probability("drop_quant_p", drop_quant_p)
        self.corrupt_tokens_p = _validate_probability(
            "corrupt_tokens_p", corrupt_tokens_p
        )
        self.min_corrupt_tokens_p = (
            self.corrupt_tokens_p
            if min_corrupt_tokens_p is None
            else _validate_probability("min_corrupt_tokens_p", min_corrupt_tokens_p)
        )
        self.apply_corrupt_tokens_p = _validate_probability(
            "apply_corrupt_tokens_p", apply_corrupt_tokens_p
        )
        if self.min_corrupt_tokens_p > self.corrupt_tokens_p:
            raise ValueError(
                "min_corrupt_tokens_p must be <= corrupt_tokens_p, got "
                f"min_corrupt_tokens_p={self.min_corrupt_tokens_p}, "
                f"corrupt_tokens_p={self.corrupt_tokens_p}."
            )

        levels_tensor = torch.tensor(levels, dtype=torch.int32)
        basis_tensor = torch.cumprod(
            torch.tensor([1, *levels[:-1]], dtype=torch.int32), dim=0
        )
        self.register_buffer("_levels", levels_tensor, persistent=False)
        self.register_buffer("_basis", basis_tensor, persistent=False)

        self.dim = len(levels)
        self.codebook_size = int(levels_tensor.prod().item())
        codebook = self.indices_to_embedding(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", codebook, persistent=False)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  levels={self._levels.tolist()!r},\n"
            f"  codebook_size={self.codebook_size!r},\n"
            f"  drop_quant_p={self.drop_quant_p!r},\n"
            ")"
        )

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Smoothly bound continuous latents into the scalar-code range."""
        levels = self._levels.to(device=z.device, dtype=z.dtype)
        half_width = (levels - 1) * (1.0 + eps) / 2.0
        offset = torch.where(levels.remainder(2) == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_width)
        return torch.tanh(z + shift) * half_width - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Return normalized quantized codes with the same shape as ``z``."""
        bounded = self.bound(z)
        drop_quant_p = self.drop_quant_p if self.training else 0.0
        quantized = round_ste_quant_dropout(bounded, drop_quant_p)
        half_width = (self._levels // 2).to(device=z.device, dtype=z.dtype)
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = (self._levels // 2).to(
            device=zhat_normalized.device, dtype=zhat_normalized.dtype
        )
        return zhat_normalized * half_width + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = (self._levels // 2).to(device=zhat.device, dtype=zhat.dtype)
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Map normalized code vectors to integer codebook indices."""
        if zhat.shape[-1] != self.dim:
            raise ValueError(
                f"expected code dimension {self.dim}, got {zhat.shape[-1]}."
            )
        shifted = torch.round(self._scale_and_shift(zhat.float())).to(torch.int64)
        basis = self._basis.to(device=zhat.device, dtype=torch.int64)
        return (shifted * basis).sum(dim=-1).to(torch.int32)

    def indices_to_embedding(self, indices: Tensor) -> Tensor:
        """Map integer codebook indices back to normalized code vectors."""
        indices = indices.to(dtype=torch.int64).unsqueeze(-1)
        basis = self._basis.to(device=indices.device, dtype=torch.int64)
        levels = self._levels.to(device=indices.device, dtype=torch.int64)
        non_centered = (indices // basis) % levels
        return self._scale_and_shift_inverse(non_centered.float())

    def corrupt_quant(self, quant: Tensor) -> Tensor:
        """Replace a random token subset with random codebook entries."""
        token_shape = quant.shape[:-1]
        random_indices = torch.randint(
            low=0,
            high=self.codebook_size,
            size=token_shape,
            device=quant.device,
        )
        random_quant = self.implicit_codebook.to(device=quant.device)[random_indices]
        sample_corrupt_p = random.uniform(
            self.min_corrupt_tokens_p, self.corrupt_tokens_p
        )
        corrupt_mask = torch.rand(token_shape, device=quant.device) < sample_corrupt_p
        return torch.where(corrupt_mask.unsqueeze(-1), random_quant, quant)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward_z(self, z: Tensor) -> tuple[Tensor, torch.LongTensor]:
        if z.shape[-1] != self.dim:
            raise ValueError(
                f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"
            )
        quant = self.quantize(z.float())
        if (
            self.training
            and self.corrupt_tokens_p > 0.0
            and random.random() < self.apply_corrupt_tokens_p
        ):
            quant = self.corrupt_quant(quant)
        tokens = self.codes_to_indices(quant).long()
        return quant, cast(torch.LongTensor, tokens)

    @torch.compiler.disable
    def forward(self, latents: Tensor) -> tuple[Tensor, torch.LongTensor]:
        if isinstance(latents, list):
            raise TypeError(
                "FiniteScalarQuantization.forward expects a tensor shaped (B, L, D), not a list."
            )
        return self.forward_z(latents)

    _levels: Tensor
    _basis: Tensor
    implicit_codebook: Tensor
