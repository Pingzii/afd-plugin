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

Build the plugin-owned operators on every node and restart AFD:

```bash
SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 pip install -e . -v --no-build-isolation
```

The loader logs the `afd-plugin` vendor library path and requires all four
`afd_ascend.afd_async_*` registrations. It does not fall back to external CAM.

## Runtime libraries cannot be loaded

If `libhccl.so` cannot be found, make sure the Ascend toolkit environment is
loaded before starting AFD. The plugin loader prepends its vendor paths and
preserves the existing CANN library paths.

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

Three cause families are worth separating before touching the model:

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
payload is therefore copied into a buffer of exactly the tile size: pad rows keep
zeros for the hidden states and the sentinel the FFN maps back to token 0 for the
ids, and the receive trims the returned tile with the reference tensor's row
count. Two steps keep the rows their forward produced instead, because their
metadata does not describe one tile for the whole step: an AFD ubatch, which
reports per-stage counts, and a rank whose counts cannot describe every Attention
rank. An Attention rank that produced more rows than the tile is the one case the
layout cannot represent, and eager steps fail fast on it.
