#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/DeepSeek-V4-Flash}"
: "${AFD_HOST:?Set AFD_HOST to this FFN node IP}"
: "${NIC_NAME:?Set NIC_NAME to the HCCL/Gloo network interface}"

AFD_PORT="${AFD_PORT:-29666}"
SERVER_PORT="${SERVER_PORT:-8901}"
TP_SIZE="${TP_SIZE:-8}"
ATTENTION_RANKS="${ATTENTION_RANKS:-8}"
FFN_RANKS="${FFN_RANKS:-8}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,afd}"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"

ADDITIONAL_CONFIG="$(printf \
  '{"afd":{"role":"ffn","connector":"CAMP2pAFDConnector","host":"%s","port":%s,"num_attention_ranks":%s,"num_ffn_ranks":%s}}' \
  "$AFD_HOST" "$AFD_PORT" "$ATTENTION_RANKS" "$FFN_RANKS")"

exec env VLLM_USE_V1=1 VLLM_USE_V2_MODEL_RUNNER=0 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$SERVER_PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --gpu-memory-utilization "${MEMORY_UTILIZATION:-0.9}" \
  --block-size 128 \
  --quantization ascend \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --served-model-name deepseek_v4_afd_ffn \
  --additional-config "$ADDITIONAL_CONFIG"
