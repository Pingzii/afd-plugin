# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Ask the device whether MC2 accepts a pre-quantized activation.

The AFD side of the MXFP8 pass-through already hands the FFN a float8_e4m3fn
activation together with a two-dimensional E8M0 scale. Whether MC2's dispatch
operator accepts that combination is a property of the installed torch-npu and
CANN build, not of this repository, so it has to be asked on the device.

Scope: this script establishes that the operator exists, that its signature takes
the arguments the dispatcher would pass, and that a call with an FP8 activation,
a two-dimensional scale and ``quant_mode=0`` completes on every rank. It does not
check that the scale is rearranged together with the activation, that the
dequantized values are correct, or that any of it survives graph capture.

Each rank prints one ``DISPATCH_COMPLETED`` line listing every output the operator
returned, because the position of the scale among them is not assumed here. A
missing operator prints ``OPERATOR_MISSING``, which is a capability gap; a
signature or argument error prints the ``SIG`` line so the two are not confused.

Run on four idle NPU devices, one process each:

    ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 HCCL_BUFFSIZE=2048 \
        timeout 180s torchrun --standalone --nproc-per-node=4 \
        tests/npu/mc2_prequant_smoke.py

A timeout is not a pass: it usually means the ranks disagree about shape or
ordering rather than that the capability is missing.
"""

import inspect
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu

WORLD_SIZE = 4
ROWS = 2
HIDDEN = 4096
MX_BLOCK_SIZE = 32
TOP_K = 2

local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
device = torch.device("npu", local_rank)

dist.init_process_group("hccl", timeout=timedelta(seconds=120))
rank = dist.get_rank()
world = dist.get_world_size()
assert world == WORLD_SIZE, f"expected {WORLD_SIZE} ranks, got {world}"


dispatch = getattr(torch_npu, "npu_moe_distribute_dispatch_v2", None)
if dispatch is None:
    # Reported separately from an argument error: this is the capability being
    # absent, and no script change can work around it.
    print(f"rank={rank} OPERATOR_MISSING", flush=True)
    raise SystemExit(3)
print(f"rank={rank} SIG {inspect.signature(dispatch)}", flush=True)

backend = dist.group.WORLD._get_backend(device)
group_name = backend.get_hccl_comm_name(rank, init_comm=True)

# Not random: a wrong scale shows up as a wrong magnitude, and every rank gets a
# different offset so a mis-routed dispatch is visible in the values.
activation = (
    torch.arange(1, ROWS * HIDDEN + 1, device=device, dtype=torch.float32)
    .reshape(ROWS, HIDDEN)
    .div(1024)
    .add(rank)
    .to(torch.bfloat16)
)

quantized, scale = torch_npu.npu_dynamic_mx_quant(
    activation,
    dst_type=torch.float8_e4m3fn,
)
if scale.dtype == torch.uint8:
    # Some builds hand the E8M0 scale back as raw bytes.
    scale = scale.reshape(ROWS, -1).contiguous().view(torch.float8_e8m0fnu)
else:
    scale = scale.reshape(ROWS, -1).contiguous()

assert quantized.dtype == torch.float8_e4m3fn, quantized.dtype
assert scale.shape == (
    ROWS,
    HIDDEN // MX_BLOCK_SIZE,
), f"unexpected scale shape {tuple(scale.shape)}"

# Two experts per rank, so the dispatch has something to rearrange.
expert_ids = torch.tensor(
    [
        [2 * rank, 2 * ((rank + 1) % world)],
        [2 * rank + 1, 2 * ((rank + 1) % world) + 1],
    ],
    dtype=torch.int32,
    device=device,
)
assert expert_ids.shape == (ROWS, TOP_K)

print(
    f"rank={rank} calling dispatch "
    f"x={tuple(quantized.shape)}/{quantized.dtype} "
    f"scale={tuple(scale.shape)}/{scale.dtype}",
    flush=True,
)

outputs = dispatch(
    x=quantized,
    expert_ids=expert_ids,
    scales=scale,
    x_active_mask=torch.ones(ROWS, dtype=torch.bool, device=device),
    expert_scales=torch.full((ROWS, TOP_K), 0.5, dtype=torch.float32, device=device),
    group_ep=group_name,
    ep_world_size=world,
    ep_rank_id=rank,
    moe_expert_num=2 * world,
    expert_shard_type=0,
    shared_expert_rank_num=0,
    # 0 states the activation is already quantized, so the operator must not
    # quantize it again and must read ``scales`` instead.
    quant_mode=0,
    global_bs=0,
    expert_token_nums_type=0,
    tp_world_size=1,
    tp_rank_id=0,
    y_dtype=torch.float8_e4m3fn,
)
torch.npu.synchronize()

parts = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
summary = " ".join(
    f"[{index}]{tuple(t.shape)}/{t.dtype}" for index, t in enumerate(parts)
)
print(f"rank={rank} DISPATCH_COMPLETED n={len(parts)} {summary}", flush=True)

dist.destroy_process_group()
