#!/usr/bin/env bash
# AFD Attention role -- DeepSeek-V4-Flash on Ascend 950 (A5), single host 4A4F.
#
# Topology : Attention DP4/TP1 on devices 0-3, FFN DP4/TP1 on devices 4-7.
# Start    : F first, then A.
# Stop     : pkill -TERM -f 'vllm serve'     # SIGTERM, never -9
#
# A2E transport quantization is ON: the attention side quantizes the FFN
# activation to MXFP8 and transports FP8 instead of BF16. Measured +34.7%
# total-token throughput on the configuration below; see README.md for the
# A/B numbers and the concurrency precondition.
#
# Load note: AFD_NPU_CAMP2P_QUANT_MODE must match on both roles.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V4-Flash model directory}"
: "${ATTN_DEVICES:=0,1,2,3}"
: "${FFN_DEVICES:=4,5,6,7}"
: "${SERVER_PORT:=18000}"
: "${FFN_PORT:=18010}"
: "${AFD_HOST:=127.0.0.1}"
: "${AFD_PORT:=6239}"
: "${ATTN_RANKS:=4}"
: "${FFN_RANKS:=4}"

# 9216 covers the 8192-token benchmark inputs plus the 32 generated tokens.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"

# max-num-seqs is PER DATA-PARALLEL RANK. With DP4 the whole box only saturates
# at max_num_seqs * 4 concurrent requests; below that the A2E quantization gain
# disappears (measured -0.3% at 8 concurrent, +34.7% at 128).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"

# Pinned at 3072: it is the largest chunk where every DP rank actually fills to
# the cap. At 4096 only the first step reaches 4096 and the rest fall to ~2500,
# which lowers throughput for BOTH baseline and quantized runs.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-3072}"

export ASCEND_RT_VISIBLE_DEVICES="$ATTN_DEVICES"
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,afd}"
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:False}"

# A2E MXFP8 transport. DIRECT stays 0 (the safe path): the FFN side has to
# dequantize once anyway for the router and shared experts, and keeping a second
# FP8 copy measured 6.8% slower than this path.
export AFD_NPU_CAMP2P_QUANT_MODE="${AFD_NPU_CAMP2P_QUANT_MODE:-4}"
export AFD_NPU_CAMP2P_MXFP8_DIRECT="${AFD_NPU_CAMP2P_MXFP8_DIRECT:-0}"

ADDITIONAL_CONFIG="$(cat <<EOF
{
  "afd": {
    "role": "attention",
    "connector": "CAMP2pAFDConnector",
    "host": "$AFD_HOST",
    "port": $AFD_PORT,
    "num_attention_ranks": $ATTN_RANKS,
    "num_ffn_ranks": $FFN_RANKS
  }
}
EOF
)"

# Capture sizes must stay well below max-num-batched-tokens: the MoE comm method
# switches to MC2/FUSED_MC2 when local tokens <= max(capture sizes), and the
# direct path only applies on the all-gather path. 128 << 3072 keeps prefill on
# all-gather while giving decode batch coverage up to max-num-seqs.
COMPILATION_CONFIG="$(cat <<EOF
{
  "cudagraph_mode": "FULL_DECODE_ONLY",
  "cudagraph_capture_sizes": [8, 16, 32, 64, 128]
}
EOF
)"

exec vllm serve "$MODEL_PATH" \
  --served-model-name DeepSeek-V4-Flash \
  --data-parallel-size 4 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --compilation-config "$COMPILATION_CONFIG" \
  --host 127.0.0.1 \
  --port "$SERVER_PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-config '{"indexer_kv_dtype":"fp8"}' \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --additional-config "$ADDITIONAL_CONFIG"
