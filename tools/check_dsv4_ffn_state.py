#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Report the deployed DSV4 state and the F-side Hash routing facts.

Run this inside the AFD FFN worker environment, once, and it answers the
questions that a stack trace cannot:

* which commit and which file the process is actually importing;
* whether the Hash-routing decisions this repository added are present;
* which MoE class the FFN role builds, and whether its gate carries a Hash table;
* which model class the plugin registers for the DSV4 architecture.

It reads the plugin source and the plugin's own module state. It does not start a
worker, load weights, or touch a device, so it is safe to run while nothing else
is using the machine.

Usage::

    python tools/check_dsv4_ffn_state.py
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DSV4_NPU_MODULE = (
    REPO_ROOT / "afd_plugin" / "model_executor" / "models" / "npu" / "deepseek_v4.py"
)
FFN_RUNNER = REPO_ROOT / "afd_plugin" / "v1" / "worker" / "npu" / "ffn_model_runner.py"

_EXPECTED_FUNCTIONS = (
    "_align_hash_table_with_ids",
    "_disable_ffn_hash_routing",
    "transport_input_ids_enabled",
)

_EXPECTED_METHODS = ("set_ffn_hash_routing",)


def _git(args: list[str]) -> str:
    # The harness may inject an incomplete GIT_CONFIG_* pair, which makes every
    # git invocation fail with "missing config key". Drop it for this call only.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_CONFIG_")
    }
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            env=env,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "<unavailable>"


def _report_location() -> None:
    print("== deployed code ==")
    print(f"repo root       : {REPO_ROOT}")
    print(f"git HEAD        : {_git(['log', '--oneline', '-1'])}")
    print(f"git status      : {_git(['status', '--short']) or '<clean>'}")
    print(f"dsv4 module     : {DSV4_NPU_MODULE}")
    print(f"dsv4 lines      : {len(DSV4_NPU_MODULE.read_text().splitlines())}")


def _report_source_surface() -> None:
    print()
    print("== source surface ==")
    source = DSV4_NPU_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in _EXPECTED_FUNCTIONS:
        print(f"{name:34s}: {'present' if name in functions else 'MISSING'}")

    # ``set_ffn_hash_routing`` is a method, so look inside the class body rather
    # than at module level.
    methods = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for node in node.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in _EXPECTED_METHODS:
        print(f"{name:34s}: {'present' if name in methods else 'MISSING'}")

    runner_source = FFN_RUNNER.read_text(encoding="utf-8")
    for needle in ("_sync_ffn_hash_routing", "recv_input_ids="):
        print(
            f"{needle:34s}: {'present' if needle in runner_source else 'MISSING'}",
        )


def _report_imported_module() -> None:
    print()
    print("== imported module ==")
    try:
        from afd_plugin.model_executor.models.npu import deepseek_v4 as module
    except Exception as exc:  # pragma: no cover - depends on the environment
        print(f"import failed   : {type(exc).__name__}: {exc}")
        return

    print(f"module file     : {module.__file__}")
    for name in _EXPECTED_FUNCTIONS:
        print(f"{name:34s}: {'bound' if hasattr(module, name) else 'MISSING'}")

    config = getattr(module, "AFDDeepseekV4ForCausalLM", None)
    if config is not None:
        print(f"registered arch : {config.__module__}.{config.__qualname__}")
        print(
            "afd_requires_input_ids"
            f"{'':13s}: {getattr(config, 'afd_requires_input_ids', '<unset>')}",
        )
        for name in _EXPECTED_METHODS:
            print(
                f"{name:34s}: {'bound' if hasattr(config, name) else 'MISSING'}",
            )

    model_cls = getattr(config, "model_cls", None) if config else None
    if model_cls is not None:
        print(f"model_cls       : {model_cls.__module__}.{model_cls.__qualname__}")

    moe = getattr(module, "native", None)
    if moe is not None:
        moe_cls = getattr(moe, "DeepseekV4MoE", None)
        print(f"ffn MoE class   : {getattr(moe_cls, '__module__', '?')}")

    layer = getattr(module, "AFDDeepseekV4DecoderLayer", None)
    if layer is not None:
        try:
            print(f"layer compute_ffn_output: {inspect.getsourcefile(layer)}")
        except TypeError:  # pragma: no cover - defensive
            print("layer compute_ffn_output: <source unavailable>")


def _report_switch() -> None:
    print()
    print("== environment ==")
    for name in ("AFD_DSV4_TRANSPORT_INPUT_IDS", "AFD_DSV4_PARAM_DUMP"):
        value = os.environ.get(name)
        print(f"{name:34s}: {'<unset>' if value is None else value!r}")


def _report_registry() -> None:
    print()
    print("== plugin registration ==")
    try:
        from vllm.model_executor.models import ModelRegistry
    except Exception as exc:  # pragma: no cover - depends on the environment
        print(f"vllm unavailable: {type(exc).__name__}: {exc}")
        return
    for arch in ("AFDDeepseekV4ForCausalLM", "DeepseekV4ForCausalLM"):
        try:
            resolved = ModelRegistry.load_model_cls(arch)
        except Exception as exc:  # pragma: no cover - depends on the environment
            print(f"{arch:28s}: {type(exc).__name__}: {exc}")
            continue
        print(f"{arch:28s}: {resolved}")


def main() -> int:
    print(f"python          : {sys.executable}")
    print(f"cwd             : {os.getcwd()}")
    _report_location()
    _report_source_surface()
    _report_imported_module()
    _report_switch()
    _report_registry()
    print()
    print(
        "If any function above reads MISSING while the git HEAD matches the "
        "expected commit, the worker is importing a different copy of the plugin "
        "than the one this script inspected.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
