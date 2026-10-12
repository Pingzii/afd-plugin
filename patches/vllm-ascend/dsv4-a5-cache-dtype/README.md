# vLLM-Ascend 本地补丁：DeepSeek-V4 在 A5 上的 KV cache dtype

## 为什么需要这个

**我们 pin 的 vllm-ascend 是 `8d4409d62`（2026-09-27），它有两个问题导致 DeepSeek-V4 在 A5 上起不来：**

```
① indexer 的 K cache dtype
   attention_config.indexer_kv_dtype 默认 "auto" → 被解析成模型 dtype（bf16）
   而 A5 的 indexer 写路径只支持 fp8
   → indexer_compress_epilog_v2 tiling 失败：
      block_stride must be greater than min block size 8448

② 主 KV cache dtype
   cache_config.cache_dtype 默认 "auto" → 同样被解析成 bf16
   而 A5 上只有 FP8 那条实现（kv_compress_epilog 的 binding 只收 E5M2/E4M3；
   bf16 那条要的 sparse_flash_mla 没有编入 ascend950 的算子集）
   → kv_compress_cache must be FP8_E5M2 or FP8_E4M3, but got c10::BFloat16
```

**版本固定，拿不到上游修复，所以在这里以 patch 形式带着。**

## 补丁内容与出处

| 顺序 | 文件 | 内容 | 出处 |
|---|---|---|---|
| 1 | `0001-indexer-kv-dtype.patch` | 解决问题 ① | **上游 PR #17635**（2026-10-08 合入 main）。我们 pin 的 commit 比它早 11 天，所以没带上 |
| 2 | `0002-cache-dtype.patch` | 解决问题 ② | **我们自己写的**。上游至今未修 |
| — | `combined.patch` | 上面两个的拼接（逐字节相同，只是合成一个文件） | 一次 `git apply` 用 |

**`combined.patch` 是给"一次打完"用的；拆分的两个请一并保留** —— 哪天升级到含 `#17635` 的 vllm-ascend，只需要 apply `0002`，而那时 `0001` 反而会冲突。

`0001` 能干净地打到 `8d4409d62` 上（`git apply --check` 退出码 0），所以直接用上游代码，不自己重写。

## 怎么用

```bash
cd /home/s00988495/AFD/vllm-ascend          # 路径按实际环境调整
PATCHES=/home/s00988495/AFD/afd-plugin/patches/vllm-ascend/dsv4-a5-cache-dtype

# 一次打完（等价于下面按顺序打两个）
git apply $PATCHES/combined.patch

# 或者按顺序打（升级过 #17635 之后只需要打 0002）
git apply $PATCHES/0001-indexer-kv-dtype.patch
git apply $PATCHES/0002-cache-dtype.patch

# 确认
git status --porcelain
# 预期：4 个文件（见下）
```

**打完后的文件清单：**

```
M  vllm_ascend/models/deepseek_v4/indexer.py              （0001）
M  tests/ut/models/test_deepseek_v4_indexer.py            （0001）
M  vllm_ascend/platform.py                                （0002）
?? tests/ut/test_platform_dsv4_cache_dtype.py             （0002）
```

**撤掉：**

```bash
cd /home/s00988495/AFD/vllm-ascend
git apply -R /home/s00988495/AFD/afd-plugin/patches/vllm-ascend/dsv4-a5-cache-dtype/0002-cache-dtype.patch
git apply -R /home/s00988495/AFD/afd-plugin/patches/vllm-ascend/dsv4-a5-cache-dtype/0001-indexer-kv-dtype.patch
rm -f tests/ut/test_platform_dsv4_cache_dtype.py     # 0002 新增的文件，-R 不会删
```

**打完补丁后，启动脚本里这两个参数就不需要了：**

```
--kv-cache-dtype fp8                        ← 不再需要
--attention-config '{"indexer_kv_dtype": "fp8"}'   ← 不再需要
```

## 验证

**打完补丁后，起服务应该看到这一行：**

```
Using fp8 KV cache for DeepSeek V4 on Ascend A5; the bf16 KV cache path is not built for ascend950.
```

**然后输出应与带参数时逐字节一致**（`temperature=0` 下已验证过）。

**单测（需要 NPU 环境，本机无法跑）：**

```bash
cd /home/s00988495/AFD/vllm-ascend
python -m pytest -sv tests/ut/test_platform_dsv4_cache_dtype.py
python -m pytest -sv tests/ut/models/test_deepseek_v4_indexer.py
```

## 上游状态

```
① indexer_kv_dtype  →  已修：#17635（2026-10-08 合入 main）
                       我们 pin 的版本没有，所以本地带着

② cache_dtype       →  未修。已确认 upstream main（090e3bbfa, 10-12）里
                       仍然没有解析它，dsa_attn_kv_plan.py:196 的 .view
                       也原样未改
                       → 待提交上游 PR
```

**哪天升级 vllm-ascend 到含 `#17635` 的版本：**

- `0001` 可以去掉
- `0002`（`cache_dtype`）仍需保留，直到上游修掉它

## 注意

**这两个 patch 是针对 `8d4409d62` 写的。** 如果哪天 vllm-ascend 不再是这个 commit，先 `git apply --check` 验证再打 —— `platform.py` 在上游变动频繁（`8d4409d62 → main` 之间改了 +285/−75），冲突是可能的。
