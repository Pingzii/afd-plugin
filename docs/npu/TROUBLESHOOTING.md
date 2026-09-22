# NPU troubleshooting

This guide covers common startup and runtime failures for the Ascend CAM
connectors. Use it together with the connector-specific setup guide:

- [CAM async connector](CAM_ASYNC_CONNECTOR_USER_GUIDE.md)
- [CAM P2P connector](CAM_P2P_CONNECTOR_USER_GUIDE.md)

Before troubleshooting, confirm that every Attention and FFN process uses the
same model, AFD topology, rendezvous address, and connector settings. Also use
the vLLM, vLLM-Ascend, CANN, and CAM versions documented by the selected
connector guide.

## CAM operators cannot be loaded

Typical errors include:

```text
aclnnCamMoeDistributeDispatchRecv not found
PTA call acl api failed
missing async CAM operator
```

Confirm that the CAM package is installed on every node and make its operator
libraries visible before starting vLLM:

```bash
CAM_VENDOR_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM
export ASCEND_CUSTOM_OPP_PATH="${CAM_VENDOR_PATH}:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${CAM_VENDOR_PATH}/op_api:${CAM_VENDOR_PATH}/op_api/lib:${LD_LIBRARY_PATH:-}"
```

The CAM `op_api/lib` directory must contain `libopapi.so`. If the package only
contains `libcust_opapi.so`, create the expected link from that directory:

```bash
cd "${CAM_VENDOR_PATH}/op_api/lib"
ln -s libcust_opapi.so libopapi.so
```

Restart every AFD process after changing the loader environment.

## Runtime libraries cannot be loaded

If `libhccl.so` cannot be found, make sure the Ascend toolkit environment is
loaded before adding the CAM paths. Always prepend to `LD_LIBRARY_PATH` as
shown above; replacing it removes the CANN library paths.

The toolkit environment does not include the NNAL ATB library. If startup
reports that `libatb.so` cannot be found, load the ATB environment after the
toolkit environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
python3 -c 'import ctypes; ctypes.CDLL("libatb.so")'
```

Prefer the checked-in launch scripts so every role receives the same library
paths.

## HCCL fails to allocate memory

An `EL0004` error during startup usually means that the vLLM memory pool left
too little device memory for HCCL. The CAM async validation recipe reserves
more memory for communication with:

```bash
export HCCL_BUFFSIZE=4096
export HCCL_OP_EXPANSION_MODE=AIV
```

and starts vLLM with:

```text
--gpu-memory-utilization 0.8
```

Start with the values in the recipe for your connector. If allocation still
fails, stop stale worker processes and reduce `--gpu-memory-utilization` before
reducing `HCCL_BUFFSIZE`.

Use `npu-smi info` to check for workers left by an earlier deployment. Only
terminate processes that belong to that deployment.

## HCCL connection times out

For `EI0006`, socket timeout, or rendezvous timeout errors, verify all of the
following:

- `host` is reachable from every participating node and `port` is free;
- `num_attention_ranks` and `num_ffn_ranks` match the processes that were
  started;
- all ranks use the same connector and HCCL settings;
- multi-node ranks have working HCCL connectivity and are placed on a
  supported network topology.

Large deployments may also need a longer connection timeout:

```bash
export HCCL_CONNECT_TIMEOUT=3600
```

Apply the same timeout to every process.

## The first CAM async request hangs

`CAMAsyncAFDConnector` requires AFD async-DP on every role:

```json
{
  "afd": {
    "connector": "CAMAsyncAFDConnector",
    "async": true,
    "compute_gate_on_attention": true
  }
}
```

For Attention data parallelism greater than one, all Attention engines must use
the AFD async-DP scheduling path. Confirm that the plugin, vLLM, and
vLLM-Ascend versions match the connector guide and are the same on every rank.
Also check that all configured Attention and FFN ranks reached connector
initialization; a missing rank prevents the CAM collective from completing.

## CAM async fails with `507015` or an AI Core timeout

Do not reduce `HCCL_BUFFSIZE` below the value used by the validated recipe. CAM
dispatch uses capacity-sized buffers, so a small request can still require the
full communication buffer.

Sequence parallel deployments can also fail intermittently when asynchronous
kernel launch is enabled. Use:

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

If the failure remains, disable sequence parallelism and verify the plain TP
topology first.

## FFN workers do not exit

An idle async FFN worker can remain blocked in `async_dispatch_recv` during
shutdown. Allow the normal service shutdown to finish before forcing process
termination. Before restarting the deployment, use `npu-smi info` to confirm
that the old workers no longer hold device memory. A process left in an
uninterruptible device wait may require the platform's NPU runtime recovery
procedure.

## Inspect async MoE split shapes

For request- or token-split ubatching, enable the shape-only diagnostic on the
Attention process:

```bash
export AFD_ASYNC_MOE_LAYOUT_LOG=1
```

The log reports token extents, padding, CAM-local slices, and FFN result
gathering. Disable it after diagnosis to keep normal service logs concise.

## Hash routing fails with an AI Core address error

A DSV4 Hash (token-keyed) layer indexes its token-to-expert table with the raw
token ids of the rows it routes, and the CANN operator never validates them. Ids
that do not belong to those rows therefore surface on device as:

```text
errorStr: The DDR address of the MTE instruction is out of range
fault kernel_name=MoeGatingTopKHash_...
```

The lines after the first `MTE`/`VEC` error are the aborted-context cascade
(`EnterFailureAbort`, `Stream Synchronize failed`, `aclnnInplaceCopy`,
`507035`), so diagnose the first error rather than the last one.

Four cause families are worth separating before touching the model:

- **Uneven Attention peers.** When there are more Attention ranks than FFN ranks,
  one FFN rank serves several Attention peers, and A2E lays that rank's ids and
  hidden states out as equal per-peer tiles. The kernel pairs an FFN rank `r` with
  Attention ranks `r, r + ffn_size, ...`, and the receiver sizes its tile as
  `batch_size / (attention_size / ffn_size)`, so peers that send different token
  counts shift every later peer's rows and leave the tail ids rows unwritten.
  `CAMP2pAFDConnector` now pads every peer of a group up to the largest count of
  that group (`afd_plugin/a2e_layout.py`) and sizes the receiver from the same
  padded tiles, so the transfer is aligned in eager steps as well as in graphs.
  DP ranks hold independent batches, which is where uneven counts come from; check
  `num_tokens_across_dp_cpu` and the TP-to-AFD rank expansion it is read through.
- **A padded step that sends unpadded rows.** When the runner pads the Attention
  batch (FULL CUDA graphs, DP padding) it reports the padded token count for the
  rank, and A2E reads exactly `batch_size / attnToMoeRatio` rows per peer, in both
  directions. A sender that writes fewer rows leaves the tail of its regions
  unwritten, and the receiver's over-read first crosses the zero-filled scales
  region and then the sender's activations, which a Hash layer reads as huge token
  ids. This is why a larger capture size faults where a smaller one does not: the
  over-read grows with the tile. `CAMP2pAFDConnector` pads the Attention payload up
  to the tile and trims the received tile back to the rows the model produced.
- **A tile that ignores the FlashComm shard.** With sequence parallelism
  (FlashComm v1, `enable_sp()`) a DP rank's rows are padded to a multiple of the
  TP size and split across its TP workers, so one Attention rank holds
  `ceil(count / tp)` rows while `num_tokens_across_dp_cpu` reports the whole DP
  rank. A tile built from the DP count is then `tp` times the payload the sender
  writes: with `4A2F`, `tp=2` and 24 tokens the sender writes 12 rows and a
  receiver sized by the DP count reads 32, so ids rows 24..31 hold that sender's
  activations -- the exact rows, and only those rows, that fault in the plog
  (rows 12..23 are the sender's zero-filled scales and stay silent, which is why
  the fault looks like an off-by-eight instead of a factor of two). Every other
  per-rank token dimension in vLLM-Ascend divides by the TP size under
  `enable_sp()` (`vllm_ascend/worker/model_runner_v1.py`, `intermediate_tokens`,
  `max_actual_tokens`), and `afd_plugin/a2e_layout.py::flash_comm_shard` now
  applies the same divisor to the tile. It is 1 when SP is off, so a run without
  FlashComm keeps its layout.
- **A receiver and a sender that derive different tiles.** The FFN rank cannot
  read the sender's row count from the transfer, so both roles have to reach the
  tile from the same inputs. A receiver that falls back to its own
  `max_num_tokens` while the sender pads to a smaller tile produces the same
  over-read as above, and with `tp>1` it produces exactly the 12-vs-32 shape.
  `attention_tile_rows` and `fallback_tile_rows` are the single derivation both
  roles call, and the `rowCount` in the faulting `MoeGatingTopKHash` tiling data
  is the tile the receiver used: `2 * tile` for `4A2F`.
- **Ids shorter than the rows the router sees.** vLLM-Ascend's fused selector
  re-aligns `forward_context.input_ids` to the MoE's sequence-parallel layout: it
  pads them to `padded_num_tokens`, splits them across the tensor-parallel group,
  and splits them again for FlashComm v1. That is correct where the MoE chunks the
  activations it routes, but the AFD FFN role routes the complete A2E tile and
  never chunks them, so the ids the connector installs already cover every router
  row and the re-alignment only shrinks the buffer below what the kernel iterates.
  The kernel then reads unrelated device memory as token ids, which is visible in
  the plog as float-pattern values (around 1e9) in the faulting cores' SU
  registers instead of token ids.
  `afd_plugin/compat/patches/npu/hash_ids_alignment.py` patches the selector so
  that ids which already describe the router rows are used unchanged. Confirm the
  patch is in the tree the run imports before attributing a fault to this family:
  it leaves the ids untouched, so a run that still faults identically after it is
  installed is faulting on what the transfer delivered, not on the re-alignment.
- **Ids that are not tokens.** Enable the value check on the FFN process:

  ```bash
  export AFD_VALIDATE_HASH_TOKEN_IDS=1
  ```

  The error names the offending row indices and values, which is what separates
  an id buffer that was never fully written from a `tid2eid` table that is
  smaller than the id space. Pad rows carry the -1 sentinel the router maps back
  to token 0, and the check skips exactly that value. Turn it off after diagnosis:
  the check reads device tensors and synchronises.

The padding is decided from `num_tokens_across_dp_cpu`, which the connector
publishes to the FFN role before every send, so both sides derive the same tile:
the sender pads up to it and the receiver sizes the operator with it. It cannot
compare the tile with the rows a traced forward produced, because that
specializes the token dimension the compiled model declares dynamic and
`torch.compile` rejects it with a constraint violation naming `input_ids`. The
control plane therefore also copies the counts into plain integers when it
publishes them, and everything the connector derives from them (the tile, the
rows an FFN rank computes on) comes from that snapshot: inside a traced forward
the DP tensors are symbolic, and comparing those symbols fails with
`Could not guard on data-dependent expression`. The payload is copied into a
buffer of exactly the tile size: pad rows keep zeros for the hidden states and the
sentinel the FFN maps back to token 0 for the ids, and the receive trims the
returned tile with the reference tensor's row count. A rank whose counts cannot
describe every Attention rank, and a topology the `attention_size // ffn_size`
layout does not describe, size the tile from `max_num_tokens` divided by the same
shard; both roles read that fallback from the same scheduler configuration, so a
step without usable metadata still keeps one tile for the whole step. An Attention
rank that produced more rows than the tile is the one case the layout cannot
represent, and eager steps fail fast on it.
