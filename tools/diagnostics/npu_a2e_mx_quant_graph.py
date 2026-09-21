# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Isolate the A2E-to-MX-quant path inside one NPU graph.

The first ``--ffn-ranks`` devices are FFN ranks and the remaining devices are
Attention ranks. Run local token counts in separate processes so a device-side
failure cannot hide which size failed. For a 4A2F topology, for example::

    ASCEND_LAUNCH_BLOCKING=1 python \
      tools/diagnostics/npu_a2e_mx_quant_graph.py \
      --devices 0,1,2,3,4,5 --ffn-ranks 2 --attention-ranks 4 \
      --local-tokens 8 --hidden-size 4096 --topk 8

On each FFN rank, A2E receives ``local_tokens * attention_ranks / ffn_ranks``
rows. Only that received tensor is passed to ``npu_dynamic_mx_quant``. This
keeps the test independent of model weights while preserving the real CAMP2P
communication boundary and graph address lifetime.
"""

from __future__ import annotations

import argparse
import os
import traceback
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from afd_plugin.compat.npu.ops import ensure_cam_p2p_ops_available

ensure_cam_p2p_ops_available()
import afd_plugin._C_ascend  # noqa: E402, F401


@dataclass(frozen=True)
class RunConfig:
    devices: tuple[int, ...]
    ffn_ranks: int
    attention_ranks: int
    local_tokens: int
    hidden_size: int
    topk: int
    aiv_num: int
    dst_type: str
    replays: int
    master_port: int


def _parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        required=True,
        help="Comma-separated physical NPU ids; FFN ranks must come first.",
    )
    parser.add_argument("--ffn-ranks", type=int, default=2)
    parser.add_argument("--attention-ranks", type=int, required=True)
    parser.add_argument("--local-tokens", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--aiv-num", type=int, default=8)
    parser.add_argument("--dst-type", choices=("fp8", "fp4"), default="fp8")
    parser.add_argument("--replays", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29611)
    args = parser.parse_args()

    devices = tuple(int(item) for item in args.devices.split(",") if item)
    world_size = args.ffn_ranks + args.attention_ranks
    if len(devices) != world_size:
        parser.error(f"--devices needs exactly {world_size} ids, got {devices}")
    if args.ffn_ranks <= 0 or args.attention_ranks <= 0:
        parser.error("rank counts must be positive")
    if args.attention_ranks % args.ffn_ranks != 0:
        parser.error("attention-ranks must be divisible by ffn-ranks")
    if min(args.local_tokens, args.hidden_size, args.topk, args.aiv_num) <= 0:
        parser.error("token, shape, topk, and aiv values must be positive")
    if args.replays < 0:
        parser.error("--replays must be non-negative")

    return RunConfig(
        devices=devices,
        ffn_ranks=args.ffn_ranks,
        attention_ranks=args.attention_ranks,
        local_tokens=args.local_tokens,
        hidden_size=args.hidden_size,
        topk=args.topk,
        aiv_num=args.aiv_num,
        dst_type=args.dst_type,
        replays=args.replays,
        master_port=args.master_port,
    )


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


def _make_inputs(
    rank: int,
    config: RunConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    is_ffn = rank < config.ffn_ranks
    if is_ffn:
        aggregate_tokens = (
            config.local_tokens * config.attention_ranks // config.ffn_ranks
        )
        return (
            torch.empty(
                (0, config.hidden_size), dtype=torch.bfloat16, device="npu"
            ),
            torch.empty((0, config.topk), dtype=torch.int32, device="npu"),
            torch.empty((0, config.topk), dtype=torch.float32, device="npu"),
            aggregate_tokens,
        )

    # Encode the source rank and row in every hidden-state row. This lets the
    # FFN side verify that graph replay preserved both peer order and payload.
    row_values = torch.arange(
        config.local_tokens,
        dtype=torch.bfloat16,
        device="npu",
    ) + rank * config.local_tokens
    hidden_states = row_values.view(-1, 1).expand(-1, config.hidden_size)
    hidden_states = hidden_states.contiguous()
    # DeepSeek-V4's attention-side gate mode transports repeated token ids in
    # the expert_ids slot and zeroes the unused scales slot.
    token_base = (rank - config.ffn_ranks) * config.local_tokens
    token_ids = torch.arange(
        token_base,
        token_base + config.local_tokens,
        dtype=torch.int32,
        device="npu",
    ).view(-1, 1)
    token_ids = token_ids.repeat(1, config.topk)
    scales = torch.zeros(
        (config.local_tokens, config.topk),
        dtype=torch.float32,
        device="npu",
    )
    return hidden_states, token_ids, scales, config.local_tokens


def _a2e_quant_step(
    *,
    rank: int,
    config: RunConfig,
    group_ep: str,
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    scales: torch.Tensor,
    operator_batch_size: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor] | None]:
    outputs = torch.ops.afd_ascend.a2e(
        hidden_states,
        token_ids,
        scales,
        operator_batch_size,
        config.hidden_size,
        config.topk,
        config.ffn_ranks,
        config.attention_ranks,
        rank,
        group_ep,
        config.aiv_num,
        1,
    )
    quantized = None
    if rank < config.ffn_ranks:
        quantized = _quantize(outputs[0], dst_type=config.dst_type)
    return outputs, quantized


def _describe_outputs(
    rank: int,
    outputs: tuple[torch.Tensor, ...],
    quantized: tuple[torch.Tensor, torch.Tensor] | None,
) -> str:
    a2e_shape = tuple(outputs[0].shape)
    a2e_ptr = outputs[0].data_ptr()
    if quantized is None:
        return f"a2e_shape={a2e_shape} a2e_ptr=0x{a2e_ptr:x}"
    value, scale = quantized
    return (
        f"a2e_shape={a2e_shape} a2e_ptr=0x{a2e_ptr:x} "
        f"quant_shape={tuple(value.shape)} scale_shape={tuple(scale.shape)} "
        f"rank={rank}"
    )


def _assert_same_tensor(
    *,
    rank: int,
    phase: str,
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual_cpu = actual.detach().cpu()
    if actual_cpu.shape == expected.shape and torch.equal(actual_cpu, expected):
        return

    mismatch = None
    if actual_cpu.shape == expected.shape:
        mismatch_indices = torch.nonzero(actual_cpu != expected, as_tuple=False)
        if mismatch_indices.numel():
            index = tuple(int(item) for item in mismatch_indices[0].tolist())
            mismatch = (
                f"first_mismatch={index} actual={actual_cpu[index]} "
                f"expected={expected[index]}"
            )
    raise AssertionError(
        f"rank={rank} phase={phase} payload={name} mismatch; "
        f"actual_shape={tuple(actual_cpu.shape)} "
        f"expected_shape={tuple(expected.shape)} {mismatch or ''}"
    )


def _validate_ffn_payload(
    *,
    rank: int,
    phase: str,
    config: RunConfig,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    if rank >= config.ffn_ranks:
        return

    source_ranks = [
        rank + (peer_index + 1) * config.ffn_ranks
        for peer_index in range(config.attention_ranks // config.ffn_ranks)
    ]
    expected_hidden_parts = []
    expected_id_parts = []
    for source_rank in source_ranks:
        row_values = (
            torch.arange(config.local_tokens, dtype=torch.bfloat16)
            + source_rank * config.local_tokens
        )
        expected_hidden_parts.append(
            row_values.view(-1, 1).expand(-1, config.hidden_size).contiguous()
        )
        token_base = (source_rank - config.ffn_ranks) * config.local_tokens
        expected_ids = torch.arange(
            token_base,
            token_base + config.local_tokens,
            dtype=torch.int32,
        ).view(-1, 1)
        expected_id_parts.append(expected_ids.repeat(1, config.topk))

    expected_hidden = torch.cat(expected_hidden_parts)
    expected_ids = torch.cat(expected_id_parts)
    expected_scales = torch.zeros_like(expected_ids, dtype=torch.float32)
    _assert_same_tensor(
        rank=rank,
        phase=phase,
        name="hidden_states",
        actual=outputs[0],
        expected=expected_hidden,
    )
    _assert_same_tensor(
        rank=rank,
        phase=phase,
        name="token_ids",
        actual=outputs[1],
        expected=expected_ids,
    )
    _assert_same_tensor(
        rank=rank,
        phase=phase,
        name="scales",
        actual=outputs[2],
        expected=expected_scales,
    )
    print(
        f"rank={rank} phase={phase} payload_validation=PASS "
        f"source_ranks={source_ranks}",
        flush=True,
    )


@torch.inference_mode()
def _run_rank(rank: int, config: RunConfig) -> None:
    phase = "initialize"
    physical_device = config.devices[rank]
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(config.master_port)
        torch.npu.set_device(physical_device)
        world_size = config.ffn_ranks + config.attention_ranks
        dist.init_process_group("hccl", rank=rank, world_size=world_size)
        ep_group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
        backend = ep_group._get_backend(torch.device("npu"))
        group_ep = backend.get_hccl_comm_name(rank)
        hidden_states, token_ids, scales, operator_batch_size = _make_inputs(
            rank, config
        )
        torch.npu.synchronize()

        role = "ffn" if rank < config.ffn_ranks else "attention"
        print(
            f"rank={rank} device={physical_device} role={role} "
            f"local_tokens={config.local_tokens} "
            f"operator_batch={operator_batch_size}",
            flush=True,
        )

        phase = "eager"
        dist.barrier()
        print(f"rank={rank} phase=eager status=BEGIN", flush=True)
        eager_outputs, eager_quantized = _a2e_quant_step(
            rank=rank,
            config=config,
            group_ep=group_ep,
            hidden_states=hidden_states,
            token_ids=token_ids,
            scales=scales,
            operator_batch_size=operator_batch_size,
        )
        torch.npu.synchronize()
        _validate_ffn_payload(
            rank=rank,
            phase=phase,
            config=config,
            outputs=eager_outputs,
        )
        print(
            f"rank={rank} phase=eager status=PASS "
            f"{_describe_outputs(rank, eager_outputs, eager_quantized)}",
            flush=True,
        )

        phase = "capture"
        dist.barrier()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        print(f"rank={rank} phase=capture status=BEGIN", flush=True)
        with torch.npu.graph(graph):
            graph_outputs, graph_quantized = _a2e_quant_step(
                rank=rank,
                config=config,
                group_ep=group_ep,
                hidden_states=hidden_states,
                token_ids=token_ids,
                scales=scales,
                operator_batch_size=operator_batch_size,
            )
        torch.npu.synchronize()
        _validate_ffn_payload(
            rank=rank,
            phase=phase,
            config=config,
            outputs=graph_outputs,
        )
        print(
            f"rank={rank} phase=capture status=PASS "
            f"{_describe_outputs(rank, graph_outputs, graph_quantized)}",
            flush=True,
        )

        for replay_index in range(config.replays):
            phase = f"replay[{replay_index}]"
            dist.barrier()
            print(
                f"rank={rank} phase=replay index={replay_index} status=BEGIN",
                flush=True,
            )
            graph.replay()
            torch.npu.synchronize()
            _validate_ffn_payload(
                rank=rank,
                phase=phase,
                config=config,
                outputs=graph_outputs,
            )
            print(
                f"rank={rank} phase=replay index={replay_index} status=PASS",
                flush=True,
            )
    except Exception:
        print(
            f"rank={rank} device={physical_device} phase={phase} status=FAIL",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    config = _parse_args()
    world_size = config.ffn_ranks + config.attention_ranks
    ffn_tokens = (
        config.local_tokens * config.attention_ranks // config.ffn_ranks
    )
    print(
        f"topology={config.attention_ranks}A{config.ffn_ranks}F "
        f"local_tokens={config.local_tokens} "
        f"ffn_tokens={ffn_tokens} "
        f"hidden={config.hidden_size} topk={config.topk} "
        f"dst={config.dst_type} devices={config.devices}",
        flush=True,
    )
    mp.spawn(_run_rank, args=(config,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
