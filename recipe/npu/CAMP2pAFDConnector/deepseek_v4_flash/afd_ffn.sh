#!/usr/bin/env bash
# AFD FFN role -- DeepSeek-V4-Flash on Ascend 950 (A5), single host 4A4F.
#
# Topology : Attention DP4/TP1 on devices 0-3, FFN DP4/TP1 on devices 4-7.
# Start    : this role first, then afd_attention.sh.
# Stop     : pkill -TERM -f 'vllm serve'     # SIGTERM, never -9
#
# Keep AFD_NPU_CAMP2P_QUANT_MODE identical to the attention role; a mismatch
# makes one side transport BF16 while the other expects FP8 + MX scale.
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

MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-3072}"

export ASCEND_RT_VISIBLE_DEVICES="$FFN_DEVICES"
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,afd}"
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:False}"

# A2E MXFP8 transport, safe path. See afd_attention.sh for why DIRECT stays 0.
export AFD_NPU_CAMP2P_QUANT_MODE="${AFD_NPU_CAMP2P_QUANT_MODE:-4}"
export AFD_NPU_CAMP2P_MXFP8_DIRECT="${AFD_NPU_CAMP2P_MXFP8_DIRECT:-0}"

ADDITIONAL_CONFIG="$(cat <<EOF
{
  "afd": {
    "role": "ffn",
    "connector": "CAMP2pAFDConnector",
    "host": "$AFD_HOST",
    "port": $AFD_PORT,
    "num_attention_ranks": $ATTN_RANKS,
    "num_ffn_ranks": $FFN_RANKS
  }
}
EOF
)"

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
  --port "$FFN_PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-config '{"indexer_kv_dtype":"fp8"}' \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --additional-config "$ADDITIONAL_CONFIG"
