# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Isolate MoeGatingTopKHash eager, NPU Graph capture, and replay.

The defaults other than vocabulary size match the DeepSeek-V4 fault kernel
decoded from an Ascend plog: float32 router logits, int64 token ids, int32
tid2eid, 256 experts, top-k 6, sqrt-softplus routing, and scaling factor 1.5.

Run one row count per process so a device exception does not hide other cases::

    ASCEND_LAUNCH_BLOCKING=0 python \
      tools/diagnostics/npu_moe_gating_topk_hash_graph.py \
      --rows 64 --vocab-size 129280 --device 0
"""

from __future__ import annotations

import argparse
import os

import torch
import torch_npu  # noqa: F401 - registers torch.npu


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--group-count", type=int, default=1)
    parser.add_argument("--topk-group", type=int, default=1)
    parser.add_argument("--routed-scaling-factor", type=float, default=1.5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--replays", type=int, default=2)
    return parser.parse_args()


def _load_ascend_ops() -> None:
    # Set the device before importing the extension so runtime initialization
    # cannot pre-empt ASCEND_RT_VISIBLE_DEVICES handling in the serving setup.
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # noqa: F401


def _run_hash(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    *,
    topk: int,
    topk_group: int,
    group_count: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops._C_ascend.moe_gating_top_k_hash(
        x=router_logits,
        k=topk,
        bias=None,
        input_ids=input_ids,
        tid2eid=tid2eid,
        k_group=topk_group,
        group_count=group_count,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-20,
        group_select_mode=1,
        renorm=0,
        norm_type=2,
        out_flag=False,
    )


def _validate_outputs(
    *,
    phase: str,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    weights: torch.Tensor,
    expert_ids: torch.Tensor,
) -> None:
    expected_ids = tid2eid.index_select(0, input_ids).detach().cpu()
    actual_ids = expert_ids.detach().cpu()
    if actual_ids.shape != expected_ids.shape or not torch.equal(
        actual_ids, expected_ids
    ):
        mismatch = None
        if actual_ids.shape == expected_ids.shape:
            indices = torch.nonzero(actual_ids != expected_ids, as_tuple=False)
            if indices.numel():
                index = tuple(int(item) for item in indices[0].tolist())
                mismatch = (
                    f"first_mismatch={index} actual={actual_ids[index]} "
                    f"expected={expected_ids[index]}"
                )
        raise AssertionError(
            f"phase={phase} expert_ids mismatch "
            f"actual_shape={tuple(actual_ids.shape)} "
            f"expected_shape={tuple(expected_ids.shape)} {mismatch or ''}"
        )
    if not bool(torch.isfinite(weights).all().item()):
        raise AssertionError(f"phase={phase} produced non-finite routing weights")
    print(
        f"phase={phase} payload_validation=PASS "
        f"weights_shape={tuple(weights.shape)} "
        f"expert_ids_shape={tuple(expert_ids.shape)}",
        flush=True,
    )


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    positive_values = (
        args.rows,
        args.vocab_size,
        args.experts,
        args.topk,
        args.group_count,
        args.topk_group,
    )
    if min(positive_values) <= 0 or args.replays < 0:
        raise ValueError("shape values must be positive and replays non-negative")
    if args.topk > args.experts:
        raise ValueError("topk cannot exceed experts")
    if args.experts % args.group_count != 0:
        raise ValueError("experts must be divisible by group-count")
    if os.environ.get("ASCEND_LAUNCH_BLOCKING", "0") == "1":
        raise RuntimeError(
            "NPU Graph is incompatible with ASCEND_LAUNCH_BLOCKING=1; "
            "unset it or set it to 0"
        )

    torch.npu.set_device(args.device)
    _load_ascend_ops()
    device = torch.device(f"npu:{args.device}")
    router_logits = torch.randn(
        (args.rows, args.experts),
        dtype=torch.float32,
        device=device,
    )
    input_ids = torch.arange(args.rows, dtype=torch.int64, device=device)
    input_ids.remainder_(args.vocab_size)
    tid2eid = torch.arange(
        args.vocab_size * args.topk,
        dtype=torch.int32,
        device=device,
    ).view(args.vocab_size, args.topk)
    tid2eid.remainder_(args.experts)
    torch.npu.synchronize()

    print(
        f"rows={args.rows} experts={args.experts} topk={args.topk} "
        f"vocab_size={args.vocab_size} input_ids_dtype={input_ids.dtype} "
        f"tid2eid_dtype={tid2eid.dtype} device={args.device}",
        flush=True,
    )
    print("phase=eager status=BEGIN", flush=True)
    eager_weights, eager_ids, _ = _run_hash(
        router_logits,
        input_ids,
        tid2eid,
        topk=args.topk,
        topk_group=args.topk_group,
        group_count=args.group_count,
        routed_scaling_factor=args.routed_scaling_factor,
    )
    torch.npu.synchronize()
    _validate_outputs(
        phase="eager",
        input_ids=input_ids,
        tid2eid=tid2eid,
        weights=eager_weights,
        expert_ids=eager_ids,
    )
    print("phase=eager status=PASS", flush=True)

    graph = torch.npu.NPUGraph()
    print("phase=capture status=BEGIN", flush=True)
    with torch.npu.graph(graph):
        graph_weights, graph_ids, _ = _run_hash(
            router_logits,
            input_ids,
            tid2eid,
            topk=args.topk,
            topk_group=args.topk_group,
            group_count=args.group_count,
            routed_scaling_factor=args.routed_scaling_factor,
        )
    torch.npu.synchronize()
    print(
        "phase=capture status=PASS payload_validation=SKIP "
        "reason=validate_after_replay",
        flush=True,
    )

    for replay_index in range(args.replays):
        phase = f"replay[{replay_index}]"
        print(f"phase={phase} status=BEGIN", flush=True)
        graph.replay()
        torch.npu.synchronize()
        _validate_outputs(
            phase=phase,
            input_ids=input_ids,
            tid2eid=tid2eid,
            weights=graph_weights,
            expert_ids=graph_ids,
        )
        print(f"phase={phase} status=PASS", flush=True)


if __name__ == "__main__":
    main()
