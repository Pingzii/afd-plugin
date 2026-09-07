# DeepSeek-V4 AFD on Ascend A5 with HCCL P2P

This is the first functional NPU path for DeepSeek-V4 AFD. It reuses the
native vLLM-Ascend DeepSeek-V4 DSA, mHC, and MoE implementation and separates
execution immediately before each decoder FFN.

The checked-in launchers default to:

```text
/mnt/share/weight/DeepSeek-V4-Flash
```

Override it with `MODEL_PATH` when required.

## Current support boundary

- Hardware: Ascend A5 (Ascend 950).
- Communication: synchronous blocking HCCL P2P through
  `CAMP2pAFDConnector`.
- Payload A to F: normalized hidden states followed by token-aligned
  `torch.int32` input IDs. The IDs are required by V4 hash routing.
- Payload F to A: FFN output hidden states.
- Gate and the complete native MoE run on the FFN role.
- Pipeline/context parallelism, sequence-parallel MoE, EPLB/elastic EP, LoRA,
  speculative decoding, and graph execution are intentionally rejected for
  this first path.
- `--enforce-eager` is mandatory while the connector uses blocking P2P.
- The reference launchers explicitly select model runner V1 for the initial
  smoke and accuracy pass.

The connector API already exposes optional `input_ids` on its backend-neutral
receive payload. When the A5 A2E/E2A operators are ready, implement the same
payload contract in the non-P2P branch of `CAMP2pAFDConnector`; the model and
FFN runner do not need another interface change.

## Reference topology

The scripts use an 8A8F reference: one 8-NPU Attention node and one 8-NPU FFN
node, each with TP8. Set `TP_SIZE`, `ATTENTION_RANKS`, `FFN_RANKS`, and
`ASCEND_RT_VISIBLE_DEVICES` to match the real cluster. The total AFD rank
counts must describe all workers across both roles, and `ATTENTION_RANKS`
must be divisible by `FFN_RANKS` for the current contiguous P2P mapping.

Install this branch in the same environment as its matching vLLM and
vLLM-Ascend checkouts:

```bash
cd /path/to/afd-plugin
pip install -e . --no-build-isolation -v
```

Start the FFN node first:

```bash
cd recipe/npu/CAMP2pAFDConnector/deepseek_v4
AFD_HOST=<FFN_NODE_IP> NIC_NAME=<HCCL_NIC> bash afd_ffn.sh
```

Then start the Attention node with the same rendezvous address and port:

```bash
cd recipe/npu/CAMP2pAFDConnector/deepseek_v4
AFD_HOST=<FFN_NODE_IP> NIC_NAME=<HCCL_NIC> bash afd_attention.sh
```

Send requests only to the Attention API port (`8900` by default). Both roles
must use identical model weights, rank counts, `AFD_HOST`, and `AFD_PORT`.

Start with a short prompt and one request. Before performance measurement,
compare greedy output against a non-AFD native vLLM-Ascend deployment using
the same checkpoint and serving parameters.
