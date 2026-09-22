# vLLM-Ascend compatibility patches

These patches adapt the vLLM-Ascend revision pinned by AFD without modifying
or publishing the dependency repository.

## CANN MoE gating TopK

`0001-use-cann-moe-gating-top-k-for-hash-routing.patch` replaces the legacy
vLLM-Ascend `MoeGatingTopKHash` custom operator with the maintained CANN
`torch_npu.npu_moe_gating_top_k` API. It is based on vLLM-Ascend commit
`80d8c194f`.

Apply it from the vLLM-Ascend repository root:

```bash
git apply --check \
  /path/to/afd-plugin/patches/vllm-ascend/0001-use-cann-moe-gating-top-k-for-hash-routing.patch
git apply \
  /path/to/afd-plugin/patches/vllm-ascend/0001-use-cann-moe-gating-top-k-for-hash-routing.patch
```

Verify the effective call site:

```bash
grep -n "npu_moe_gating_top_k" \
  vllm_ascend/ops/fused_moe/experts_selector.py
```

To remove the patch:

```bash
git apply -R \
  /path/to/afd-plugin/patches/vllm-ascend/0001-use-cann-moe-gating-top-k-for-hash-routing.patch
```
