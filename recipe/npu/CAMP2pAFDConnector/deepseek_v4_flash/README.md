# DeepSeek-V4-Flash on A5 with A2E MXFP8 transport quantization

Single host, 8× Ascend 950 (`Ascend950DT_9574`), 4A4F: attention DP4/TP1 on
devices 0-3, FFN DP4/TP1 on devices 4-7, connector `CAMP2pAFDConnector`.

The attention side quantizes the FFN activation to MXFP8 (E4M3 + E8M0 block
scale) and transports FP8 instead of BF16, which halves the A2E payload.

## Configurations

| Piece | Value |
| --- | --- |
| `AFD_NPU_CAMP2P_QUANT_MODE` | `4` (MXFP8) on **both** roles |
| `AFD_NPU_CAMP2P_MXFP8_DIRECT` | `0` — safe path, see "Why not the direct path" |
| `--max-num-batched-tokens` | `3072` |
| `--max-num-seqs` | `128` |
| `--compilation-config` | `cudagraph_mode=FULL_DECODE_ONLY`, `cudagraph_capture_sizes=[8,16,32,64,128]` |
| `--max-model-len` | `9216` |
| `kv-cache-dtype` | `fp8`, `attention-config={"indexer_kv_dtype":"fp8"}` |
| `HCCL_BUFFSIZE` | `2048` |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:False` |

## Measured (128 concurrent requests, 4×8192 in / 32 out, two runs each)

| Run | `mode 0` (BF16 transport) | MXFP8 safe path (`DIRECT=0`) | Δ |
| --- | --- | --- | --- |
| chunk 3072 | 236,378 token/s | **318,519 token/s** | **+34.7%** |
| chunk 4096 | 218,646 token/s | 297,245 token/s | +35.9% |

TTFT median 3708 ms → 2995 ms (−19%). All runs: `Failed Requests = 0`.

GSM8K: 93.86% (`mode 0`) vs **93.71%** (safe path), i.e. about −0.15 points, with
roughly ±0.5 points of measurement spread on this benchmark (the same path
measured 92.87% once in an earlier configuration — do not claim lossless).

## Two preconditions, both easy to get wrong

**1. Concurrency must saturate all four DP ranks.**
`--max-num-seqs` is a *per-rank* limit, so a DP4 box only saturates at
`max_num_seqs × 4 = 128` in-flight requests. With 8 concurrent requests the same
quantized path measured **−0.3%** instead of +34.7%: the machine is idle enough
that the transport hides behind compute. Every performance number taken before
this was understood (including −12.7% and −5.7% conclusions) was measured on a
partially idle box.

**2. `--max-num-batched-tokens 3072`, not 4096.**
At 3072 all four ranks fill to the cap. At 4096 only the first step reaches 4096
and later steps fall to ~2500, which costs throughput for *both* the baseline and
the quantized run; the ratio stays ~35% but the absolute numbers drop.

## Why not the direct path (`DIRECT=1`)

The project originally aimed to have the FFN side consume the transported FP8
without re-quantizing. Measured, that is a **net loss**: 296,934 token/s vs
318,519 (−6.8%).

Reason: two FFN consumers need a model-dtype activation regardless — the router
(`MoeGatingTopKHash`, `x=router_logits`) and the shared experts (they quantize
their input unconditionally). So the dequantize happens anyway. The direct path
adds a second live copy of the activation plus one extra DP `all_gather` per
layer for the MX scale, and only saves the MoE's internal quantization
(~37 MB/layer) — less than it costs.

`AFD_NPU_CAMP2P_MXFP8_DIRECT` therefore defaults to `0` and should stay there.

The direct path also hits an operator limit that the safe path never triggers:
`MoeGatingTopKHash` faults with `aivec error 263` once the gathered row count
reaches ~14,288 (chunk ≈ 3,570). Reproduced deterministically, with all Python
inputs verified (dtype, contiguity, row count, finiteness, token-id range,
`tid2eid` shape, memory fragmentation). Not localized to a source line; treat as
an operator-side defect and keep it out of the supported configuration.

## Reproducing

```bash
export MODEL_PATH=/path/to/DeepSeek-V4-Flash

# FFN first, then attention
bash afd_ffn.sh > /tmp/serve_F.log 2>&1 &
sleep 30
bash afd_attention.sh > /tmp/serve_A.log 2>&1 &
```

Verify which path is active before trusting a benchmark number — the two are easy
to confuse:

```bash
grep -c "consumed directly" /tmp/serve_F.log      # >0 = direct, 0 = safe path
grep -c "dequantized for the token dispatch" /tmp/serve_F.log
```

Tear down with `pkill -TERM -f 'vllm serve'`; give the NPUs a moment before
starting a new run, otherwise the next process sees the devices as busy.
