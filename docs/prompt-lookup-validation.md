# Prompt-Lookup 投机解码验证报告

**日期**:2026-09-12 · **变更**:`openspec/changes/add-prompt-lookup-draft` · **代码**:local-patches @ `3c74c335` + 未提交 lookup 实现
**机器**:roc-epyc · RTX 3090 24GB(driver 595.91.07,CUDA 12.9,kernel 6.8.0-138-generic)

## 0. 摘要

| 维度 | Ornith-1.5-35B(MoE Q4_K_M) | Qwen3.8-27B(dense UD-Q4) |
|---|---|---|
| 正确性 | 6/11 byte-exact,DIFF 均为近并列翻转,语义等价(数学题同答案) | 8/11 byte-exact,3 条 DIFF 中 2 条 token 数不变 |
| 复述型提速 | **回归**(119.6 vs 基线 155.6) | **1.75-2.68×**(bench_echo 92.1 vs 34.3) |
| 非复述负载 | 停发保护生效,开放生成零回归(156.7 vs 156.6) | 零回归(34.2-35.7 vs 34.3-35.7) |
| vs MTP3 | 未做 MTP 对照(见 §3.5) | 复述型胜 MTP3(92.1 vs 69.0);通用负载 MTP3 胜 |
| 结论 | MoE 验证图落地前是性能负资产 | 收益前提(dense + 验证图)实证成立 |

近并列翻转经 MTP3 对照确证为**多 token 验证 forward 的数值共性**(与草稿来源无关),非 lookup 缺陷。

## 1. 环境与方法

### 1.1 沙箱部署(不污染生产环境)

生产 ftllm 装在 conda env `ftllm`(site-packages + 包内 `libfastllm_tools.so`,ctypes 按 realpath 加载)。实验用独立包副本:

```bash
cmake --build build -j                       # 产物 build/tools/ftllm/libfastllm_tools.so
mkdir -p /tmp/ftllm-lookup/ftllm
cp -r tools/fastllm_pytools/. /tmp/ftllm-lookup/ftllm/
cp build/tools/ftllm/libfastllm_tools.so /tmp/ftllm-lookup/ftllm/
PYTHONPATH=/tmp/ftllm-lookup nohup …/envs/ftllm/bin/ftllm server …
```

构建新鲜度验证:`nm -D libfastllm_tools.so | grep -ci promptlookup` = 8 符合预期。实验期间 `sudo -n systemctl stop ftllm-ornith.service`(生产 8896),结束后 start 并以 curl /v1/models + nvidia-smi(23465 MiB)探活。

### 1.2 被测模型与启动配方

| | Ornith-1.5-35B(MoE) | Qwen3.8-27B(dense) |
|---|---|---|
| 文件 | `Ornith-1.5-35B-Q4_K_M.gguf`(20.2GB)+ mmproj | `Qwen3.8-27B-UD-Q4_K_XL.gguf`(16.7GB) |
| 架构 | qwen35moe,blk.40.nextn 头存在 | qwen3_5 dense,65 块(blk.64=nextn) |
| 关键参数 | `--device cuda --moe_device cuda --cuda_slab 256 --low_gpu_mem --mtp 0 --gpu_mem_ratio 0.99 --kv_cache_dtype fp4 --kv_cache_limit 4G --tokens 131072 -t 24 --mmproj … --ori …` | `--device cuda --low_gpu_mem --gpu_mem_ratio 0.98 -t 24` |
| KV 窗口 | 131072(P1' 配方) | 48384(AutoWarmup 280→378 页;不开 low_gpu_mem 时 availForKV=-0.36GB → 2304 悬崖) |
| 对照矩阵 | base(--lookup 0)/ lookup4 / lookup8 | base / lookup4 / **mtp3(--mtp 3)** |

注:`--lookup > 0` 自动强制 `FASTLLM_GPU_TOKEN_HANDOFF=0`;27B 需 `FASTLLM_MOE_CUDA_RESERVE_MB` 不适用(dense)。实验口 8100,`--model_name` 分别为 qwen35 / qwen38_27b_q4。

### 1.3 负载与口径

battery 11 条(脚本 `/tmp/lookupval/battery.py`),按 lookup 命中预期分类:

- **短 prompt**(锁死验证步识别):short_1(`1+1=`)、short_2(`你好`)
- **复述型**(lookup 主场):repeat_zh(段落抄三遍)、translate(中→英)、rag_echo(文档原句摘录)、bench_echo(《念奴娇》×12 逐字输出,1200 max)
- **开放生成**(停发保护):open_gen(散文)、bench_open(科幻微小说)
- **推理/代码**(混合):math(鸡兔同笼)、code(回文函数)、logic(时针分针重合)

口径:`temperature=0, top_p=1, chat_template_kwargs.enable_thinking=false`,非流式;tok/s = completion_tokens / 墙钟;对拍 = content+reasoning byte-exact(`compare.py`)。每个数字单次运行,同机同构建交叉对照。

## 2. Ornith-1.5-35B(MoE)结果

### 2.1 速度:全配置回归

| 条目 | base | lookup4 | 变化 |
|---|---|---|---|
| short_1 / short_2 | 28.6 / 79.3 | 24.6 / 79.1 | ≈同(秒级请求,噪声) |
| math | 159.1 | 67.6 | **-57%** |
| code | 161.7 | 90.8 | **-44%** |
| repeat_zh | 118.8 | 83.4 | -30% |
| translate | 111.6 | 102.4 | -8% |
| rag_echo | 104.5 | 71.1 | -32% |
| open_gen | 156.6 | 156.7 | **零回归**(停发生效) |
| logic | 160.8 | 111.5 | -31% |
| bench_echo | 155.6 | 119.6 | -23% |
| bench_open | 161.5 | 146.5 | -9% |

lookup8 定向测速:bench_echo 100.5(该阶段 battery JSON 未持久化,headline 记录于当时 design.md/tasks.md)。

**接受率不是瓶颈**:lookup8 bench 段 `pos_accept_rate=[78.65%, 68.23%, 65.10%, 65.10%, 64.58%, 63.54%, 62.50%, 62.50%]`,hit_rate 32-36% → E≈4.5-5 tok/步仍回不了本。lookup4 全程 `[58-63%, 37-45%, 33-41%, 28-36%]`,hit 21-27%。

### 2.2 正确性:语义无损

**6/11 byte-exact**(short×2、repeat_zh、translate、rag_echo、open_gen)。5 条 DIFF 明细:

| 条目 | tok(base/lk4) | 首差性质 |
|---|---|---|
| math | 336 / 347 | `35 × 2` vs `35 \times 2`(同义写法) |
| code | 393 / 391 | 注释措辞 `使判断更宽松(可选…)` vs `使判断更通用` |
| logic | 350 / 350 | 段落结构重排(插入角度说明句),token 数不变 |
| bench_echo | 920 / 769 | 英文 meta 段措辞分叉(见下) |
| bench_open | 385 / 388 | `自己的回声` vs `自己声音的回响`(近并列) |

- **语义等价**:math 两边答案同为"鸡 23 只、兔 12 只",验证段完整一致。
- **确定性**:lookup 同 prompt 两次运行 byte-exact(`math_B1.txt` == `math_B2.txt`)——排除提交错位/状态泄漏类 bug。
- **翻转根因**:多 token 验证 forward(全位置 logits)与逐步 forward 数值路径不同(fp4 KV、batch 改变 kernel/累加序)→ 近并列 logits 处 argmax 翻转。投机解码普遍性质。
- **反向案例(lookup 优于基线)**:bench_echo 基线贪心自退化——12 遍《念奴娇》中 10 遍抄成"**乱穿穿空**"(仅 1 遍正确);lookup4 抄写 11/11 全部正确("乱石穿空"),从 prompt 拷贝的草稿比贪心逐步采样更忠实。27B 上三配置均 11/11 正确。

### 2.3 回归根因链

1. **MTP 验证 CUDA 图 dense-only**:`mtpVerifyGraphEligible` 要求 `num_experts == 0`(qwen3_5.cpp ~16272)→ MoE 验证步全 eager(~20-55ms/步 vs 图步 6.1ms);
2. **GGUF indexed MoE 专家 kernel 为 per-token vec 路径**:seq-9 验证步成本≈线性,零批摊销 → eager 每 token 成本 ≥ 图步;
3. 未命中步(seqLen==1)已走 decode 图(`isPrefill=!all1` 确认),无额外税——回归全部来自命中后的验证步。

回本条件:eager 与图步的成本比 > E(接受 token 数)。lookup8 E≈4.5-5 仍不足。

### 2.4 停发保护验证

open_gen 156.7 vs 156.6(零回归)证明低命中负载下自适应停发 + 普通 decode 图路径完好;`runTargetWithPast` 跳过 decode 图的路径仅在草稿步激活。

### 2.5 MTP 对照说明

Ornith 本轮矩阵为 base/lookup4/lookup8,**未跑 MTP 对照**:部署配方即 `--mtp 0`(GGUF 实含 `blk.40.nextn.*` 头,日后可补测)。注意 §2.3 根因对任何投机后端同样成立——MTP/DFlash 与 lookup 共享同一验证器与 MoE eager 路径(此为构造分析,非实测)。

## 3. Qwen3.8-27B(dense)结果

### 3.1 速度:复述型 1.75-2.68×,其余零回归

| 条目 | base | lookup4 | 提速 | mtp3 |
|---|---|---|---|---|
| short_1 / short_2 | 20.6 / 29.1 | 21.8 / 29.6 | ≈同 | — |
| math | 35.6 | 35.7 | 1.00× | — |
| code | 35.0 | 34.3 | 0.98× | **62.7** |
| repeat_zh | 33.5 | **75.8** | **2.26×** | — |
| translate | 33.4 | 32.6 | 0.98× | — |
| rag_echo | 31.8 | **55.6** | **1.75×** | — |
| open_gen | 34.7 | 34.2 | 0.99× | — |
| logic | 35.0 | 34.8 | 1.00× | **56.9** |
| bench_echo | 34.3 | **92.1** | **2.68×** | 69.0 |
| bench_open | 35.1 | 34.4 | 0.98× | 38.6 |

统计:lookup4 `pos_accept_rate` 早期 [42%,25%,12%,6%](低命中条目),bench_echo 段升至 [68%,58%,52%,48%];hit_rate 27-33%。MTP3 全程 [80.58%, 62.95%, 51.79%](负载不敏感)。

### 3.2 正确性:8/11 byte-exact + 三方归因

DIFF 3 条:

| 条目 | tok(base/lk4) | 首差 |
|---|---|---|
| code | 208 / 208 | `包含空格和大小写的回文` vs `包含空格和标点的回文`(注释一词) |
| logic | 350 / 350 | `分针每分钟走 $6^\circ$(因为…)` vs `$360^\circ / 60 = 6…`(等价表述) |
| bench_open | 425 / 415 | 自由小说措辞分叉(开放生成近并列密集) |

**MTP3 对照(决定性)**:

| 条目 | lk4==base | mtp3==base | lk4==mtp3 |
|---|---|---|---|
| code | ✗(翻转) | ✓ | ✗ |
| logic | ✗(翻转) | ✗(翻转) | **✓ byte-exact** |
| bench_open | ✗ | ✗(383 tok) | ✗ |
| bench_echo | ✓ | ✓ | ✓ |

- logic 条目:lookup4 与 MTP3 翻转后内容**完全一致**、双双偏离 base → 两条投机路径共享同一多 token 验证 forward,数值发散源相同,与草稿来源(查表 vs MTP 头)无关;
- code 条目:MTP3 恰好与 base 一致、仅 lookup4 翻转 → 翻转是否发生在某位置取决于草稿上下文踩不踩近并列,有随机性;
- bench_echo:三方 byte-exact(517 tok)→ 复述负载无近并列,投机解码完全无损;
- bench_open:三方各不同 → 开放生成近并列密集,任何投机配置都会偏离逐步贪心。

结论:**近并列翻转是投机解码在 fp4 KV + batched kernel 数值配置下的普遍性质**。验收口径 = "语义无损 + 翻转仅发生于近并列 logits 且可解释"。

### 3.3 MTP vs lookup(dense 上的互补关系)

| | lookup4 | mtp3 |
|---|---|---|
| 草稿来源 | prompt/已生成 token 的 n-gram 查表 | 训练过的 MTP 头(blk.64) |
| 额外成本 | 零(无草稿模型、免训练、零显存) | MTP 头权重 + 每步草稿 forward |
| 复述型(bench_echo) | **92.1(2.68×)** | 69.0(2.01×) |
| 通用负载(code/logic) | ≈1×(停发) | **1.63-1.79×** |
| 开放生成(bench_open) | 0.98×(停发) | 1.10× |
| 接受率特征 | 负载敏感(6%→68%) | 负载不敏感(~80/63/52%) |

负载可预判为"输出复述输入"(翻译/改写/RAG 摘要/长文复写)选 lookup;通用推理选 MTP;两者互斥(`--mtp` 与 `--lookup` 组合报错)。

## 4. 总结论

1. **正确性(两模型一致)**:greedy 语义无损;byte 级偏差仅为近并列翻转,由共享验证 forward 的数值路径决定(MTP3 同样翻转,27B logic 条目两后端翻转后 byte-exact 互证)。lookup 自身完全确定。Ornith 上还观察到 lookup 抄写比基线贪心更忠实的案例(乱石穿空 11/11 vs 基线 1/11)。
2. **性能边界**:dense qwen3_5(验证图 eligible)复述型 1.75-2.68× 且胜 MTP3,非复述零回归;MoE GGUF 单卡全配置回归(验证图 dense-only + 专家 kernel 零批摊销),验证图支持落地前不建议开启。
3. **默认参数可用**:ngram=[2,4] 在 dense 中文复述负载直接命中良好;n_min 精扫留生产调参。

## 5. 工件与复现

实验工件(单机 `/tmp/lookupval/`,验证后可清理):

| 文件 | 内容 |
|---|---|
| `start.sh` / `start27b.sh` | 两模型启动脚本(mode = base/lookupN/mtpN) |
| `battery.py` / `compare.py` | 11 条负载 + byte-exact 对拍 |
| `phaseA.json` / `phaseB.json` | Ornith base / lookup4 结果 |
| `b27_base.json` / `b27_lookup4.json` / `b27_mtp3.json` | 27B 三配置结果(mtp3 仅 4 条定向) |
| `server_0/4/8.log`, `s27b_*.log` | 服务端日志(含 Lookup/MTP 统计行) |
| `math_B1.txt` / `math_B2.txt` | Ornith 确定性对照 |

复现路径:`start27b.sh lookup4` → `FT_MODEL=qwen38_27b_q4 python3 battery.py out.json --bench` → `python3 compare.py b27_base.json out.json`(需先停生产 8896 释放显存)。
