#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run the external-routing comparison on the FFN workers, without A or an HTTP server.

Deliberately not ``vllm serve``: the serving path calls ``initialize_from_config``,
which unconditionally initializes the KV cache, initializes the A-to-F connector, and
starts the FFN server loop. The connector initialization is a handshake with the
Attention side, so a run with no A service would block there, and the server loop would
compete with the verification for the same connector.

This launch path stops after the model is loaded:

    worker construction -> init_device -> load_model -> verify -> shutdown

The CAMP2P connector object is still built by the runner; its constructor does not
connect, so leaving it uninitialized is safe.

Scope, fixed on purpose: BF16 activations, DP4, TP1, PCP1, no sequence parallelism,
ALLGATHER, one MoE layer (the first one unless ``--layer`` is given). No MXFP8, no MC2,
no serving trigger.

Exit codes come from ``afd_plugin.verification.launcher``. A caller-side ``timeout``
reports 124, which is never a pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from afd_plugin.verification.launcher import (
    DEFAULT_RESULTS_ROOT,
    resolve_output_dir,
    resolve_run_id,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model path or repository id")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="model layer index; defaults to the first MoE layer",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"results root; defaults to {DEFAULT_RESULTS_ROOT}/<run-id>",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="shared run id; generated once and reused by every rank",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="hidden width; read from the loaded config when omitted",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    run_id = resolve_run_id(arguments.run_id)
    output_dir = resolve_output_dir(arguments.output_dir, run_id, root=REPO_ROOT)
    raise SystemExit(
        "run_routing_verify has no device bootstrap yet.\n"
        f"  model:      {arguments.model}\n"
        f"  layer:      {arguments.layer if arguments.layer is not None else 'first MoE'}\n"
        f"  run id:     {run_id}\n"
        f"  output dir: {output_dir}\n"
        "The launcher-side pieces exist and are tested; constructing the FFN worker "
        "without initialize_from_config has not been written or run on a device.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
