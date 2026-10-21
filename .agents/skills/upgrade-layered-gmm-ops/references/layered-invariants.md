# Layered Invariants

These rules define the AFD A3 contract. A new upstream implementation may
change how the computation is organized, but it must preserve these behaviors
unless the migration explicitly stops for a contract decision.

1. `layer_index` is a required device tensor with dtype `INT64` and exactly one
   logical element. Validate it at the ACLNN entry with `GetViewShape()` or
   logical element count. Do not assert the tiling storage shape: CANN can expose
   `[1]` as storage shape `[1, 1]`.

2. Every `all_*` dynamic list has one element per layer. Each element contains
   the complete expert set for that layer. The lists used by an operator must
   have equal lengths. Expert count comes from the per-layer tensor dimension,
   never from the list length.

3. Tiling carries the list length as `layerNum`. Set it in every path that
   produces tiling data, including early-return A8W4/pregroup paths and fused
   swiglu paths. An unset value defaults dangerously.

4. The kernel reads `layer_index` once at initialization with
   `LayeredReadLayerIndex`. It resolves per-layer descriptors with
   `GetLayerTensorAddr(curLayer, layerNum, pointer)`. The helper clamps an
   invalid device index to a valid list entry instead of relying on a device
   assertion.

5. Layer selection applies to weights, bias, scales, auxiliary matrices, and
   other per-layer descriptors. It does not apply to per-call `x`, `y`,
   `group_list`, or `per_token_scale` buffers.

6. The layered shape is always the single full-expert weight case. Keep
   `isSingleWeight_ = true` and route infershape checks through the per-layer
   full-expert branch.

7. The registered node and tiling names must agree. For plain GMM this means
   `GroupedMatmulLayered` in the op definition, tiling-data registration,
   `IMPL_OP_OPTILING`, and kernel/config registration. Apply the equivalent
   `_layered` identity to swiglu.

8. Generated dtype macros follow the schema names. If an input becomes
   `all_weight`, all `DTYPE_WEIGHT`, `ORIG_DTYPE_WEIGHT`, and `FORMAT_WEIGHT`
   references that describe that input must become their `ALL_WEIGHT` forms.

The key implementation anchors are:

| Contract | Current anchor |
|---|---|
| device layer read and bounded pointer lookup | `csrc/npu/ascend_kernels/utils/op_kernel/layered/layered_tensor_addr.h` |
| plain GMM layer validation | `grouped_matmul_layered/op_api/aclnn_grouped_matmul_layered.cpp:CheckLayerIndex` |
| plain GMM list length and layerNum | `grouped_matmul_layered/op_host/op_tiling/grouped_matmul_tiling.cpp:GMMTiling::Init` |
| plain GMM single-weight semantics | `grouped_matmul_layered/op_host/op_tiling/grouped_matmul_tiling.cpp` (`isSingleWeight_ = true`) |
| plain GMM kernel entry | `grouped_matmul_layered/op_kernel/grouped_matmul_layered.cpp` |
| plain GMM descriptor addressing | `grouped_matmul_layered/op_kernel/grouped_matmul.h` and quantized kernel headers |
| fused swiglu layer validation | `grouped_matmul_swiglu_quant_v2_layered/op_api/aclnn_grouped_matmul_swiglu_quant_v2_layered.cpp` |
| fused swiglu layerNum | `grouped_matmul_swiglu_quant_v2_layered/op_host/op_tiling/grouped_matmul_swiglu_quant_v2_layered_base_tiling.cpp` |
| fused swiglu descriptor addressing | `grouped_matmul_swiglu_quant_v2_layered/op_kernel/grouped_matmul_swiglu_quant_spilit_fusion.h` and pipeline headers |
