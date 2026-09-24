# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Schedule helpers for OAT-family learned action codecs."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from itertools import accumulate

import torch

__all__ = [
    "create_register_schedule_mask",
    "is_power_of_two",
    "normalize_decode_keep_k",
    "pow2_register_schedule",
    "powers_of_two",
    "register_schedule_prefix_boundaries",
    "unit_register_schedule",
]


@lru_cache(maxsize=128)
def is_power_of_two(value: int) -> bool:
    n = int(value)
    return n > 0 and (n & (n - 1)) == 0


@lru_cache(maxsize=128)
def powers_of_two(start: int, end: int) -> tuple[int, ...]:
    lo = int(start)
    hi = int(end)
    if lo <= 0 or hi < lo:
        return ()
    return tuple(2**i for i in range(lo.bit_length() - 1, hi.bit_length()))


def unit_register_schedule(latent_horizon: int) -> tuple[int, ...]:
    """Return OAT's unit register schedule."""
    horizon = int(latent_horizon)
    if horizon <= 0:
        raise ValueError(f"latent_horizon must be positive, got {latent_horizon}.")
    return (1,) * horizon


def pow2_register_schedule(latent_horizon: int) -> tuple[int, ...]:
    """Return BOAT's coarse-to-fine power-of-two register schedule."""
    horizon = int(latent_horizon)
    if not is_power_of_two(horizon):
        raise ValueError(
            "BOAT requires latent_horizon to be a positive power of 2, "
            f"got {latent_horizon}."
        )

    schedule: list[int] = []
    remaining = horizon
    next_block = 1
    while remaining > 0:
        block = min(next_block, remaining)
        schedule.append(block)
        remaining -= block
        if len(schedule) == 1:
            next_block = 1
        else:
            next_block *= 2
    return tuple(schedule)


def _normalize_register_schedule(
    register_schedule: Sequence[int],
) -> tuple[int, ...]:
    schedule = tuple(int(v) for v in register_schedule)
    if not schedule:
        raise ValueError("register_schedule must be non-empty.")
    if any(v <= 0 for v in schedule):
        raise ValueError(f"register_schedule values must be positive, got {schedule}.")
    return schedule


def register_schedule_prefix_boundaries(
    register_schedule: Sequence[int],
) -> tuple[int, ...]:
    """Return valid cumulative prefix lengths for a register schedule."""
    return tuple(accumulate(_normalize_register_schedule(register_schedule)))


def normalize_decode_keep_k(
    decode_keep_k: Sequence[int] | None,
    *,
    batch_size: int,
    latent_horizon: int,
    default_keep_k: int,
) -> list[int]:
    """Validate or synthesize per-sample latent prefix lengths."""
    keep_k = (
        [int(default_keep_k)] * int(batch_size)
        if decode_keep_k is None
        else [int(k) for k in decode_keep_k]
    )
    if len(keep_k) != batch_size:
        raise ValueError(
            f"decode_keep_k length {len(keep_k)} must match batch size {batch_size}."
        )
    if any(k < 0 for k in keep_k):
        raise ValueError(f"All decode_keep_k values must be >= 0, got {keep_k}.")
    if any(k > latent_horizon for k in keep_k):
        raise ValueError(
            f"All decode_keep_k values must be <= latent_horizon={latent_horizon}, got {keep_k}."
        )
    return keep_k


def create_register_schedule_mask(
    register_schedule: Sequence[int],
    device: str | torch.device,
) -> torch.Tensor:
    """Create an OAT-family register attention mask.

    A register attends to every earlier schedule level and to itself. It does
    not attend to other registers in its own level. For OAT's unit schedule,
    this reduces to a standard causal lower-triangular mask. For BOAT's
    coarse-to-fine schedule, this yields blocked within-level attention.
    """
    return _create_register_schedule_mask_cached(
        _normalize_register_schedule(register_schedule), str(device)
    )


@lru_cache(maxsize=128)
def _create_register_schedule_mask_cached(
    register_schedule: tuple[int, ...],
    device: str,
) -> torch.Tensor:
    level_ids = torch.repeat_interleave(
        torch.arange(len(register_schedule), dtype=torch.long, device=device),
        torch.as_tensor(register_schedule, dtype=torch.long, device=device),
    )
    positions = torch.arange(
        int(sum(register_schedule)), dtype=torch.long, device=device
    )
    earlier_levels = level_ids.unsqueeze(0) < level_ids.unsqueeze(1)
    same_register = positions.unsqueeze(0) == positions.unsqueeze(1)
    return earlier_levels | same_register
