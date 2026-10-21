# Migration Traps

- Do not validate `layer_index` storage rank in tiling. The logical `[1]`
  tensor can be exposed as storage `[1, 1]`.
- Do not leave a dropped input as a null local passed to old validation or shape
  code. Delete or rewrite the whole obsolete path; a later upstream branch can
  make a null dereference reachable.
- Do not forget the A8W4 early-return path when setting `layerNum`. The kernel
  otherwise sees zero and the shared helper clamps every layer to layer zero.
- Do not use the dynamic list length as the expert count. It is the layer count;
  expert count comes from each layer tensor's expert dimension.
- Do not replace only ordinary `DTYPE_*` macros. Generated `ORIG_DTYPE_*` and
  `FORMAT_*` macros must follow the renamed schema input too.
- Do not leave `REGISTER_TILING_DATA_CLASS(GroupedMatmul, ...)` or
  `IMPL_OP_OPTILING(GroupedMatmul)` after renaming the node. The registry lookup
  uses the actual node type.
- Do not assume the default generated swiglu config filename matches the
  `_layered` node name. Keep the explicit `grouped_matmul_swiglu_quant_v2_layered.json`
  registration required by the current CANN version.
- Do not assert runtime correctness with identical layer tensors. Use distinct
  layer data and a nonzero `layer_index`.
- Do not treat a Meta dispatcher pass as kernel correctness. Meta never reaches
  ACLNN validation or the device kernel.
- Do not include A5 `arch35` sources as an incidental upstream refresh. A5 has a
  separate kernel entry, tiling registry, and integration contract.
- Do not preserve formatting churn as if it were semantic migration work. Keep
  the new upstream formatting where practical and inspect normalized diffs.
- Do not use the upstream C++ unit tests as proof that AFD's pybind schema and
  package registration are correct. They test a different build surface.
- Do not assume the AFD filenames rename the operator token. The gmm tree keeps
  upstream filenames for every `op_host`/`op_kernel` file except the kernel
  entry; only the swiglu tree renames the token in its paths. A wrong path map
  reports real files as absent and skips the anchor check on them.
