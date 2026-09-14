#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Print the DSV4 AFD parameter layout for the current worker role.

Purpose
-------
``load_weights`` fails with ``KeyError: 'model.layers.0.mlp.gate.tid2eid'`` when
the checkpoint carries a parameter the role-aware model never registered. That
KeyError names the missing parameter but not which role, which layer indices
count as Hash layers, or whether the MoE even built a gate.

Run this inside one AFD worker to see the registered names directly.
The intended use is at the point where the model exists but weight loading has
not started. Two drop-in options:

1. Import it from a model wrapper and call :func:`log_dsv4_parameter_layout`
   right before ``super().load_weights(...)``.
2. Attach it as a breakpoint-style hook by setting, in the worker environment::

       AFD_DSV4_PARAM_DUMP=1

   and calling :func:`maybe_log_dsv4_parameter_layout` from the AFD DSV4
   loader, which prints the same summary only when that variable is set.

Free functions only, so this can be imported from a worker process without
pulling in the plugin package.
"""

from __future__ import annotations

import os

PARAM_DUMP_ENV = "AFD_DSV4_PARAM_DUMP"
_GATE_MARKERS = ("gate.", "tid2eid")


def _iter_named_parameters(model: object):
    named = getattr(model, "named_parameters", None)
    if named is None:
        return []
    return list(named())


def log_dsv4_parameter_layout(model: object, *, role: str = "unknown") -> str:
    """Return a human-readable summary of DSV4 gate/MoE parameter names.

    Args:
        model: The causal-LM module whose parameters the loader will index.
        role: AFD role label, used only for the printed header.

    Returns:
        The summary text. Callers normally print it.
    """

    params = _iter_named_parameters(model)
    names = [name for name, _ in params]
    gate_names = sorted(n for n in names if any(m in n for m in _GATE_MARKERS))
    tid_names = sorted(n for n in names if "tid2eid" in n)
    moe_names = sorted(n for n in names if ".mlp." in n and "experts" in n)

    hash_layers = sorted(
        {
            int(n.split(".layers.")[1].split(".")[0])
            for n in tid_names
            if ".layers." in n
        },
    )
    gate_layers = sorted(
        {
            int(n.split(".layers.")[1].split(".")[0])
            for n in gate_names
            if ".layers." in n
        },
    )

    config = getattr(getattr(model, "config", None), "num_hash_layers", None)
    if config is None:
        model_cfg = getattr(getattr(model, "model", None), "config", None)
        config = getattr(model_cfg, "num_hash_layers", None)

    lines = [
        f"=== AFD DSV4 parameter layout (role={role}) ===",
        f"total registered parameters        : {len(names)}",
        f"config.num_hash_layers             : {config!r}",
        f"layers registering gate.tid2eid    : {hash_layers}",
        f"layers registering any gate param  : {gate_layers}",
        f"expert parameter count             : {len(moe_names)}",
        "",
        "--- names containing 'gate.' or 'tid2eid' (first 40) ---",
    ]
    lines.extend(f"  {n}" for n in gate_names[:40])
    if not gate_names:
        lines.append("  <none>")
    lines.append("")
    lines.append("--- Hash-layer MoE type, if reachable ---")
    for layer_idx in hash_layers[:4]:
        layer = _find_layer(model, layer_idx)
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        tid = getattr(gate, "tid2eid", "<no gate>")
        lines.append(
            f"  layer {layer_idx}: mlp={type(mlp).__name__} "
            f"gate={type(gate).__name__} "
            f"tid2eid={'None' if tid is None else type(tid).__name__}",
        )
    if not hash_layers:
        lines.append("  <no layer registered tid2eid>")
    return "\n".join(lines)


def _find_layer(model: object, layer_idx: int):
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return None
    try:
        return layers[layer_idx]
    except (IndexError, KeyError, TypeError):
        return None


def maybe_log_dsv4_parameter_layout(model: object, *, role: str = "unknown") -> None:
    """Print the layout only when ``AFD_DSV4_PARAM_DUMP`` is set."""

    if os.environ.get(PARAM_DUMP_ENV, "") not in {"1", "true", "yes", "on"}:
        return
    print(log_dsv4_parameter_layout(model, role=role), flush=True)


def log_role_filtered_names(
    weights: object,
    *,
    role: str,
    limit: int = 40,
) -> None:
    """Print which checkpoint names survive the role filter and why.

    Paired with :func:`maybe_log_dsv4_parameter_layout`: that side shows what the
    model registered, this side shows what the loader is about to feed it. A name
    that appears here but not there is the mismatch that raises ``KeyError``.
    """

    from afd_plugin.model_executor.models.npu.deepseek_v4 import (
        _checkpoint_weight_roles,
    )

    lines = [f"--- role-filtered checkpoint names (role={role}) ---"]
    kept = 0
    for index, (name, _weight) in enumerate(weights):
        owners = sorted(_checkpoint_weight_roles(name))
        if role in owners:
            kept += 1
            if kept <= limit:
                lines.append(f"  KEEP {name}  owners={owners}")
        elif index < limit:
            lines.append(f"  drop {name}  owners={owners}")
    lines.append(f"kept {kept} names for role {role!r}")
    print("\n".join(lines), flush=True)


__all__ = [
    "PARAM_DUMP_ENV",
    "log_dsv4_parameter_layout",
    "log_role_filtered_names",
    "maybe_log_dsv4_parameter_layout",
]
