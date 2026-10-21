# Rename And Trim Map

Apply the rename map after taking the new upstream source and before replaying
semantic patches. Verify generated macro names after every schema rename.

The rename map above is for file **contents**. File **paths** are asymmetric
between the two operators:

- `grouped_matmul_layered` keeps every upstream filename except the kernel
  entry `op_kernel/grouped_matmul.cpp` -> `op_kernel/grouped_matmul_layered.cpp`
  and the consolidated `op_api/` files (`op_host/op_api/*` moves to `op_api/`
  with the merges recorded in the classifier's explicit map).
- `grouped_matmul_swiglu_quant_v2_layered` renames the operator token in every
  remaining path.

Do not rename gmm `op_host`/`op_kernel` filenames when replaying; a wrong path
map makes trimmed files look absent and silently skips layered-anchor checks.

| Upstream | AFD A3 layered |
|---|---|
| `grouped_matmul` | `grouped_matmul_layered` |
| `grouped_matmul_swiglu_quant_v2` | `grouped_matmul_swiglu_quant_v2_layered` |
| `GroupedMatmul` | `GroupedMatmulLayered` |
| `GroupedMatmulSwigluQuantV2` | `GroupedMatmulSwigluQuantV2Layered` |
| `weight` / `weight_scale` / `weight_assist_matrix` | `all_weight` / `all_weight_scale` / `all_weight_assist_matrix` |
| `DTYPE_WEIGHT`, `ORIG_DTYPE_WEIGHT`, `FORMAT_WEIGHT` | `DTYPE_ALL_WEIGHT`, `ORIG_DTYPE_ALL_WEIGHT`, `FORMAT_ALL_WEIGHT` |
| `DTYPE_SCALE`, `ORIG_DTYPE_SCALE`, `FORMAT_SCALE` | corresponding `ALL_SCALE` forms where the schema input is `all_scale` |
| `DTYPE_BIAS`, `ORIG_DTYPE_BIAS`, `FORMAT_BIAS` | corresponding `ALL_BIAS` forms where the schema input is `all_bias` |
| upstream `op_host/op_api/` | AFD operator `op_api/` |

The fixed A3 input layout is:

```text
GMM: x, all_weight, all_bias, all_scale, layer_index,
     group_list, per_token_scale, y
W13: x, x_scale, group_list, layer_index, all_weight,
     all_weight_scale, all_weight_assist_matrix, bias, smooth_scale,
     y, y_scale
```

Keep ACLNN `DFX_IN`, `OP_INPUT`, pybind schemas, kernel argument order, and
op-definition indices synchronized with this order.

The A3 trim policy is explicit:

- retain `ascend910_93` / 910C configuration only;
- exclude `ascend950`, `ascend910b`, `arch35`, `apt.cpp`, and A5 tiling trees;
- exclude GMM V2-V5, WeightNz, 950 checker, and fallback variants unless the
  current AFD contract explicitly exposes them;
- exclude GMM `offset`, `antiquant_scale`, and `antiquant_offset` inputs;
- retain only the A3 dtype/format columns exercised by the current AFD tests;
- keep AFD's consolidated ACLNN L0/L2 wrappers and pybind adapters.

Do not infer that a new upstream file belongs in A3 merely because it is under
one of the two source directories. Classify it, then either add it to the
explicit A3 scope or record why it is excluded.
