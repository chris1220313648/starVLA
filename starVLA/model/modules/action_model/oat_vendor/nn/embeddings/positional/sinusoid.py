# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed sinusoid absolute position embeddings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SinusoidPosEmbed",
    "build_1d_sinusoid_pos_embed",
    "build_2d_sinusoid_pos_embed",
]

ScalingMode = Literal["absolute", "interpolate"]


def _build_interleaved_sincos(
    pos: torch.Tensor, dim: int, temperature: float
) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"dim must be even for sin/cos interleaving, got {dim}.")
    scale = -math.log(temperature) / dim
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=pos.device, dtype=pos.dtype) * scale
    )
    angles = pos.unsqueeze(-1) * div_term
    emb = torch.empty(*angles.shape[:-1], dim, device=pos.device, dtype=pos.dtype)
    emb[..., 0::2] = torch.sin(angles)
    emb[..., 1::2] = torch.cos(angles)
    return emb


def _validate_floating_dtype(dtype: torch.dtype) -> None:
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError(
            f"dtype must be floating point for sinusoid embeddings, got {dtype}."
        )


def build_1d_sinusoid_pos_embed(
    length: int,
    dim: int = 1024,
    *,
    temperature: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Interleaved sin/cos position embeddings for 1D sequences.

    Returns tensor shaped `[1, L, D]`.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    _validate_floating_dtype(dtype)
    pos = torch.arange(length, device=device, dtype=dtype)
    return _build_interleaved_sincos(pos, dim, temperature).unsqueeze(0)


def build_2d_sinusoid_pos_embed(
    height: int,
    width: int,
    dim: int = 1024,
    *,
    temperature: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Interleaved sin/cos position embeddings for 2D grids.

    Returns tensor shaped `[1, H, W, D]`.
    """
    if height <= 0 or width <= 0:
        raise ValueError(
            f"height and width must be positive, got {height} and {width}."
        )
    if dim % 4 != 0:
        raise ValueError(
            f"dim must be divisible by 4 for 2D sinusoid embedding, got {dim}."
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    _validate_floating_dtype(dtype)

    axis_dim = dim // 2
    h_pos = torch.arange(height, device=device, dtype=dtype)
    w_pos = torch.arange(width, device=device, dtype=dtype)
    h_emb = _build_interleaved_sincos(h_pos, axis_dim, temperature)
    w_emb = _build_interleaved_sincos(w_pos, axis_dim, temperature)

    h_grid = h_emb[:, None, :].expand(height, width, axis_dim)
    w_grid = w_emb[None, :, :].expand(height, width, axis_dim)
    emb = torch.cat([h_grid, w_grid], dim=-1)
    return emb.unsqueeze(0)


class SinusoidPosEmbed(nn.Module):
    """Fixed sinusoid absolute position embeddings shaped `[1, ..., D]`."""

    def __init__(
        self,
        dim: int,
        max_shape: Sequence[int],
        *,
        scaling: ScalingMode = "absolute",
        temperature: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        if not max_shape or any(size <= 0 for size in max_shape):
            raise ValueError(f"max_shape must contain positive sizes, got {max_shape}.")
        if scaling not in ("absolute", "interpolate"):
            raise ValueError(f"Unsupported scaling mode {scaling!r}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.dim = int(dim)
        self.max_shape = tuple(int(size) for size in max_shape)
        self.scaling: ScalingMode = scaling
        self.temperature = float(temperature)
        self.pos_embed: torch.Tensor
        self.register_buffer(
            "pos_embed",
            self._build(self.max_shape),
            persistent=True,
        )

    def _build(self, shape: Sequence[int]) -> torch.Tensor:
        if len(shape) == 1:
            return build_1d_sinusoid_pos_embed(
                shape[0], dim=self.dim, temperature=self.temperature
            )
        if len(shape) == 2:
            return build_2d_sinusoid_pos_embed(
                shape[0],
                shape[1],
                dim=self.dim,
                temperature=self.temperature,
            )
        raise NotImplementedError(
            "SinusoidPosEmbed supports only 1D/2D absolute position embeddings."
        )

    @torch.compiler.disable
    def forward(
        self,
        shape: Sequence[int],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        shape = tuple(int(size) for size in shape)
        if len(shape) != len(self.max_shape):
            raise ValueError(
                f"shape must have {len(self.max_shape)} dims, got {shape}."
            )
        if any(size <= 0 for size in shape):
            raise ValueError(f"shape must contain positive sizes, got {shape}.")

        if self.scaling == "absolute":
            if any(
                size > max_size
                for size, max_size in zip(shape, self.max_shape, strict=True)
            ):
                raise ValueError(
                    f"Requested shape {shape} exceeds max_shape {self.max_shape}."
                )
            slices = (slice(None), *[slice(0, size) for size in shape], slice(None))
            pos_embed = self.pos_embed[slices]
        elif self.scaling == "interpolate":
            channel_first = self.pos_embed.movedim(-1, 1)
            if len(shape) == 1:
                pos_embed = F.interpolate(
                    channel_first, size=shape[0], mode="linear", align_corners=True
                )
            elif len(shape) == 2:
                pos_embed = F.interpolate(
                    channel_first, size=shape, mode="bilinear", align_corners=True
                )
            else:
                raise NotImplementedError(
                    "SinusoidPosEmbed interpolation supports only 1D/2D shapes."
                )
            pos_embed = pos_embed.movedim(1, -1)
        else:
            raise AssertionError(f"Unhandled scaling mode {self.scaling!r}.")

        if device is not None or dtype is not None:
            return pos_embed.to(
                device=device or pos_embed.device,
                dtype=dtype or pos_embed.dtype,
            )
        return pos_embed
