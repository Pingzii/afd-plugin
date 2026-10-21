# P0-1 / P0-2 探针运行说明（临时分支）

**分支用途**：只观测 Ascend MoE 通信选路，**不改任何计算、张量或精度**。测完即可丢弃。

## 这个分支做了什么

- 新增 `afd_plugin/compat/npu/debug_moe_comm_probe.py`：打印 MoE 通信选路决策。
- `afd_plugin/compat/npu/runtime.py` 加了 3 行调用。

**默认关闭**：`AFD_DEBUG_MOE_COMM` 未设为 `1` 时，探针连 `vllm_ascend` 都不 import，行为与改动前完全一致。

## 跑之前确认

```bash
git -C <afd-plugin> log --oneline -1          # 应该是本分支的探针提交
git -C <afd-plugin> status --porcelain        # 应该为空
python -c "import afd_plugin; print(afd_plugin.__file__)"   # 确认用的是这份代码
```

## 运行

```bash
export AFD_DEBUG_MOE_COMM=1
# 照常启动（128 并发 / 8192 in / 32 out / max_num_batched_tokens=3072）
```

## 需要回收的内容

每个不同的 `(num_tokens, 选路)` 组合只打印一次，形如：

```
[AFD-P0] num_tokens=3072 capacity=128 policy=MoECommPolicy.CAPACITY_AND_WORLD_SIZE type=MoECommType.ALLGATHER impl=AllGatherCommImpl
```

**只要这些 `[AFD-P0]` 行，不需要完整日志。** 需要判断的点：

| 字段 | 用途 |
|---|---|
| `capacity` | 是否为 128（来自 `max_cudagraph_capture_size`）；实际值是多少 |
| `policy` | 是否为 `CAPACITY_AND_WORLD_SIZE` |
| `num_tokens` | prefill / decode / warmup / capture 各自的值 |
| `type` 与 `impl` | 两者是否一致（`ALLGATHER`↔`AllGatherCommImpl`，`MC2`↔`MC2CommImpl`） |

**观测到与预期不同同样是有价值的结果**，不要为了"符合预期"调整实验。

## 已知限制

- **graph replay 不会重跑 Python，因此 replay 阶段不打印。** 每个 batch size 在 eager 或 capture 时会至少出现一次。
- 一行 `[AFD-P0]` 都没有 → 环境变量没传进 server 进程（只设在外层 shell 不算）。

## 回滚

```bash
git -C <afd-plugin> checkout -- afd_plugin/compat/npu/runtime.py
rm -f <afd-plugin>/afd_plugin/compat/npu/debug_moe_comm_probe.py
```

或直接切回原分支：`git -C <afd-plugin> switch feat/a2e-attention-gate`

## 本分支**不**包含

`force_allgather`（P0-3）。它与本次观测无关，混在一起会让测量不干净。
