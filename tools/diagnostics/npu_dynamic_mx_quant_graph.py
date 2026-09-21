# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Isolate DynamicMxQuant eager, graph-capture, and replay execution on one NPU.

Run one token count per process so a device-side failure does not hide the
result for the other sizes. For example:

    ASCEND_LAUNCH_BLOCKING=1 python \
      tools/diagnostics/npu_dynamic_mx_quant_graph.py \
      --tokens 32 --hidden-size 7168 --dst-type fp8
"""

from __future__ import annotations

import argparse

import torch
import torch_npu


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dst-type", choices=("fp8", "fp4"), default="fp8")
    parser.add_argument("--replays", type=int, default=2)
    return parser.parse_args()


def _quantize(
    value: torch.Tensor,
    *,
    dst_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dst_type == "fp4":
        return torch_npu.npu_dynamic_mx_quant(
            value,
            dst_type=torch_npu.float4_e2m1fn_x2,
            round_mode="round",
        )
    return torch_npu.npu_dynamic_mx_quant(
        value,
        dst_type=torch.float8_e4m3fn,
    )


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    if args.tokens <= 0 or args.hidden_size <= 0 or args.replays < 0:
        raise ValueError("tokens/hidden-size must be positive and replays non-negative")

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    value = torch.randn(
        (args.tokens, args.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    torch.npu.synchronize()

    print(
        f"phase=eager tokens={args.tokens} hidden={args.hidden_size} "
        f"dst={args.dst_type} input_ptr=0x{value.data_ptr():x}",
        flush=True,
    )
    eager_output, eager_scale = _quantize(value, dst_type=args.dst_type)
    torch.npu.synchronize()
    print(
        f"phase=eager status=PASS output_shape={tuple(eager_output.shape)} "
        f"scale_shape={tuple(eager_scale.shape)}",
        flush=True,
    )

    graph = torch.npu.NPUGraph()
    print("phase=capture status=BEGIN", flush=True)
    with torch.npu.graph(graph):
        graph_output, graph_scale = _quantize(value, dst_type=args.dst_type)
    torch.npu.synchronize()
    print(
        f"phase=capture status=PASS output_shape={tuple(graph_output.shape)} "
        f"scale_shape={tuple(graph_scale.shape)}",
        flush=True,
    )

    for replay_idx in range(args.replays):
        print(f"phase=replay index={replay_idx} status=BEGIN", flush=True)
        graph.replay()
        torch.npu.synchronize()
        print(f"phase=replay index={replay_idx} status=PASS", flush=True)


if __name__ == "__main__":
    main()
