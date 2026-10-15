# vLLM-Ascend 本地补丁：让 FFN 消费 A2E MXFP8 送来的激活与 scale

## 为什么需要这个

**A2E 的 MXFP8 传输（afd-plugin PR #444）把 Attention 激活以 E4M3 + E8M0 MX scale 搬到 FFN。
但 FFN 侧的 MoE 会在 `prepare` 阶段**再量化一次**，把已经量化的激活当成 BF16 重新算 scale。**

```
A2E 送来:  hidden = float8_e4m3fn [tokens, hidden]
           x_scale = float8_e8m0fnu [tokens, ceil(hidden/32)]

FFN 侧原样跑:
  prepare_finalize._prepare_with_ep_group
    → torch_npu.npu_dynamic_mx_quant(hidden, dst_type=float8_e4m3fn, ...)
    → 对【已经量化的张量】再量化一次 → 数值被静默改变
    → 而且 A2E 送来的 x_scale 被完全丢弃
```

**症状会是精度掉，而不是报错 —— 所以必须显式处理。**

## 补丁内容与出处

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | `0001-a2e-mxfp8-ffn-scale.patch` | 两处改动，见下 |

**两处改动：**

```
① vllm_ascend/ops/fused_moe/prepare_finalize.py
   _prepare_with_ep_group 的 W8A8MXFP/W4A8MXFP 分支加判断：
   输入已经是 float8_e4m3fn 时，跳过 npu_dynamic_mx_quant，pertoken_scale 留 None

② vllm_ascend/ops/fused_moe/routed_experts.py
   把 self.ascend_pertoken_scale = pertoken_scale
   改成 if pertoken_scale is not None: 才赋值
   → 这样上游生产者（AFD）预置在 layer 上的 scale 不会被 None 覆盖
```

**出处：我们自己写的。上游没有这个改动** —— 因为上游还没有 A2E MXFP8 这条 AFD 专用通路。

**配套的 AFD 侧改动在 `afd-plugin` 的 `feat/npu-dsv4-a5-cache-dtype` 分支：**

```
- connector 侧：quant_mode=4 的放行、量化、补齐、传参（camp2p.py）
- FFN 侧：把收到的 x_scale 经 compute_ffn_output 装到 layer 上（ffn_model_runner.py / npu/deepseek_v4.py）
- 算子侧：afd-plugin PR #444 的 cherry-pick
```

**三者缺一不可。**

## 与已有补丁的关系

**与 `patches/vllm-ascend/dsv4-a5-cache-dtype/` **互不影响**：**
**改的文件不重叠（那边是 `indexer.py` / `platform.py`），apply 顺序随意。**

## 怎么用

```bash
cd /home/s00988495/AFD/vllm-ascend          # 路径按实际环境调整
PATCHES=/home/s00988495/AFD/afd-plugin/patches/vllm-ascend/a2e-mxfp8-ffn-scale

git apply --check $PATCHES/0001-a2e-mxfp8-ffn-scale.patch   # 先验证
git apply         $PATCHES/0001-a2e-mxfp8-ffn-scale.patch

git status --porcelain
# 预期 2 个文件：
#   M vllm_ascend/ops/fused_moe/prepare_finalize.py
#   M vllm_ascend/ops/fused_moe/routed_experts.py
```

**撤掉：**

```bash
git apply -R $PATCHES/0001-a2e-mxfp8-ffn-scale.patch
```

## 验证

**已做的（蓝区，静态）：**

```
✅ 基线：vllm-ascend 8d4409d62（env/versions.json 的 exact pin）
✅ git apply --check 退出码 0（两个文件都干净）
✅ 改动文件在 8d4409d62 与本地 HEAD 之间的差异已确认：
     prepare_finalize.py  完全相同 → 改动可复用
     routed_experts.py    有差异（+41/−5），但赋值处上下文相同（L702-703）→ 改动可复用
```

**未做的（绿区，需要 NPU）：**

```
❌ 没有实机验证过 MXFP8 通路的数值正确性
❌ 没有验证过 npu_dynamic_mx_quant 在 Attention 侧与 FFN 侧用同一套 scale 语义
❌ 没有跑过端到端精度
```

**绿区怎么验：**

```
① 打完 patch，用 quant_mode=4 起服务，确认启动无报错
② 打 N=32 并发 max_tokens=512 的负载，确认不 hang、不报 shape 错
③ 精度：GSM8K（干净基线是 93.86%，见 status/current.md），
   温度 0、max_out_len ≥ 2048、MAX_MODEL_LEN ≥ 8192
   → 如果明显掉，先怀疑 scale 语义不匹配（而不是算子）
④ 性能：对比 quant_mode=0 的每层时延
```

## 风险与已知不确定

```
① scale 语义必须两边同源
   AFD 侧调 mx_quantize_a2e_activation() → npu_dynamic_mx_quant(..., scale_alg=get_dynamic_mx_quant_scale_alg())
   FFN 侧原本也是同一个 alg
   → 但如果真机上两边对 E8M0 的解释不同，症状是精度掉而不是报错

② routed_experts 的 finally 会把 ascend_pertoken_scale 清成 None
   → AFD 必须【每一步、每一层】重新装一次，不能只装一次

③ 这个 patch 是针对 8d4409d62 写的
   → vllm-ascend 升级后先 git apply --check，两个文件都是上游高频改动区
```

## 上游状态

```
A2E 算子侧  →  vllm-project/afd-plugin PR #444（作者 wangyibo1005），
               未合入 main。本分支已 cherry-pick（提交 aa9326b）
FFN 消费侧  →  上游没有。等 A2E MXFP8 端到端打通并有实测数据后，
               再考虑按上游的方式重做（可能要走 patch/ 目录而不是直接改这两个函数）
```

**哪天 vllm-ascend 上游自己支持了"已量化激活 + 外部 MX scale"的入参：**

```
- 本 patch 可以去掉
- afd-plugin 侧改成通过上游接口传 scale
```
