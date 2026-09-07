# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Backend-neutral DeepSeek-V4 checkpoint ownership helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_BOTH_ROLES = frozenset(("attention", "ffn"))


def _weight_layer_path(name: str) -> tuple[int, str] | None:
    """Extract the decoder layer index and first layer-local path component."""
    parts = name.split(".")
    for marker_idx, part in enumerate(parts[:-2]):
        if part != "layers":
            continue
        try:
            layer_idx = int(parts[marker_idx + 1])
        except ValueError:
            continue
        return layer_idx, parts[marker_idx + 2]
    return None


def _checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Classify a native GPU or Ascend V4 checkpoint path by AFD owner."""
    if name in {
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
        "model.hc_head_fn",
        "model.hc_head_base",
        "model.hc_head_scale",
    }:
        return _ATTENTION_ROLE
    layer_path = _weight_layer_path(name)
    if layer_path is None:
        return _BOTH_ROLES
    _, stage = layer_path
    if stage in {"ffn", "mlp"}:
        return _FFN_ROLE
    return _ATTENTION_ROLE


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain only role-owned paths."""
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(name):
            yield name, loaded_weight


__all__ = ["_checkpoint_weight_roles", "_iter_role_weights"]
