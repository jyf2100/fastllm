# MTP 投机解码验证报告(Qwen3.8-27B,3090)

**日期**:2026-09-12(历史部分含 2026-09-08 实测) · **代码**:local-patches @ `3c74c335`(MTP 修复含 `RemapQwen35GGUFMtpTask`,上游 PR #721)
**机器**:roc-epyc · RTX 3090 24GB(driver 595.91.07,CUDA 12.9)
**本轮关联变更**:`openspec/changes/add-prompt-lookup-draft`(MTP3 作为 lookup 对照后端,共享验证器)

## 0. 摘要

| 维度 | 结论 |
|---|---|
| 可用性 | 3090 上 Qwen3.8-27B MTP 唯一可行路径 = **UD-Q4_K_XL GGUF 整卡驻留 + ForceFP16 remap 补丁**;FP8 分层模式无 MTP;Flash-Next FP8 上 warmup 直接崩(issue #717) |
| 速度(dense 27B) | 通用负载 **1.63-1.79×**(code 62.7 / logic 56.9 vs 基线 35),复述型 2.01×(69.0),开放生成 1.10×(38.6) |
| 接受率 | 负载不敏感,稳态 pos_accept_rate ≈ [77-81%, 57-68%, 42-52%](greedy exact) |
| 正确性 | greedy 下与逐 token 解码在近并列 logits 处同样翻转(与 lookup4 在 logic 条目翻转后 byte-exact 互证)——投机验证 forward 数值共性,非 MTP 缺陷 |
| 资源代价 | MTP 头 FP16 ~850MB + 1 lookahead 页:同配方 KV 窗口 48384 → **27392**(lookup/base) |
| vs lookup | 通用负载 MTP 胜;复述型被 lookup 反超(92.1 vs 69.0);二者互斥,按负载选 |

## 1. MTP 机制要点(fastllm 实现)

- **草稿头**:模型自带的 nextn/MTP 层(Qwen3.8-27B 为 blk.64,`qwen35moe.nextn_predict_layers=1`),`--mtp N` 控制每步草稿数;权重缺失/未融合时 `HasMtpWeights()` 失败 → 静默不启用。
- **验证器**:与 DFlash / prompt-lookup 共用同一多 token 验证机制(`Qwen35MTPForward` / runTargetWithPast);验证 CUDA 图由 `mtpVerifyGraphEligible` 门控,**要求 `num_experts == 0`(dense-only)**。
- **接受判据双模式**:greedy = exact 精确比较;采样 = typical(posterior_threshold=0.09, posterior_alpha=0.30)。enabled 日志原文:
  `[Qwen3.5 MTP] enabled: layers=1, drafts_per_step=3, root_device=cuda:0, tp_devices=1, acceptance=exact(greedy)/typical(posterior_threshold=0.09, posterior_alpha=0.30)(sampling), log_interval=64 validations.`
- **调度**:单 lane(`Fastllm Scheduler: Qwen3.5 MTP (lane=1)`),多请求串行轮转——与 lookup 相同。
- **前置条件**:模型必须整卡驻留(`CanUseGPUForward()` 门控)——分层/CPU offload 模式下 MTP 路径不存在。

## 2. 历史验证脉络(2026-09-08 → 09-12)

### 2.1 可用性排雷(按时间序)

1. **Qwen3.8-27B-FP8 分层模式**:`Qwen35MTPForward` 调用点被 `CanUseGPUForward()` 门控,分层部署日志中根本不出现 `[Qwen3.5 MTP] enabled:` → **FP8+MTP 需 ≥32GB 显存整卡**,3090 不可行,只能走 Q4 GGUF。
2. **UD 动态量化 GGUF 静默杀死 MTP**:UD-Q4_K_XL 的 65 层 ffn_gate=IQ4_XS + ffn_up=Q4_K 混合类型,`model.cpp` 合并权重要求两输入 ggmlType 相同 → gate/up 永不融合;主层支持分离投影所以模型能跑,但 `HasMtpWeights()` 硬性要求融合的 `mtp.layers.0.mlp.gateup_proj.weight` → **MTP 静默失效,无任何报错**。
   - **修复**:`RemapQwen35GGUFMtpTask`(model.cpp ~3492)对 2-D MTP 张量设 `GGUFWeightReplaceForceFP16`(反量化 FP16,单层 ~850MB),合并机制即可产出 gateup_proj + mergeqkv;`FASTLLM_DEBUG_MTP=1` 可打印 mtp.* 权重清单。已提上游 **PR #721**。
3. **Qwen3.8-Flash-Next FP8**:`--mtp>0` 时 warmup 直接 `cublas error`(2D copy invalid value → cublas MatMulTransB),FP8/FP16 draft 头都一样且不自动回退 → 上游 **issue #717**;该机以 `--mtp 0` 运行。

### 2.2 性能与标定(修复后)

| 配方 | ratio | KV 窗口 | 速度 |
|---|---|---|---|
| 2026-09-08 UD-Q4 GGUF `--mtp 3` | 0.9 | 9216 tok(72 页) | decode ~44 tok/s,pos_accept_rate [81.25%, 59.38%, 34.38%] |
| 2026-09-12 复测 | 0.95 | 17024 tok(182 页) | 与 0.9 无本质差异(速度本随接受率漂) |
| 2026-09-12 本轮 `low_gpu_mem + 0.98` | 0.98 | **27392 tok**(214+1 页) | 见 §3 |

**tok/s 波动真因 = MTP 接受率,非引擎/配置回归**:MTP 块节拍(一次 draft×3 + verify 迭代)稳定 ~46-52ms,`tok/s = E[每块接受数] ÷ 节拍`;实测接受率从 [93, 85, 80]%(E≈3.35,~72 tok/s)漂到 [73, 47, 28]%(E≈1.9,~37 tok/s),同进程同配置同 GPU 满频。接受率随生成内容与采样判据变:greedy 走 exact、采样走 typical(高熵中文文本下 typical 可比 exact 多接受,temp0 37.5 < temp0.7 46 tok/s 实测)。**跨次测速必须同 prompt 同采样参数,否则接受率噪声淹没一切;跨进程比 tok/s 先比 pos_accept_rate。**

**"流式 50ms 节拍"证伪**(受控 A/B,同 prompt 背靠背):stream 51.8 vs non-stream 49.2 tok/s;raw-socket+NODELAY 逐 chunk 时间戳显示每 SSE chunk = 一个 MTP 块(2-4 token),投递与生成 1:1,无节流。曾误判"流式硬节拍"实为低接受率阶段的生成速度 + 口径混叠。

## 3. 本轮 MTP3 对照实测(2026-09-12,lookup 验证的对照组)

### 3.1 目的与设置

为给 prompt-lookup 验证提供 MTP 基准与**翻转归因对照**(同一验证 forward 下,MTP 草稿是否同样引发近并列翻转)。

```bash
/tmp/lookupval/start27b.sh mtp3     # = ftllm server Qwen3.8-27B-UD-Q4_K_XL.gguf
                                    #   --device cuda --low_gpu_mem --gpu_mem_ratio 0.98
                                    #   --mtp 3 --model_name qwen38_27b_q4 --port 8100 -t 24
```

API 就绪 ~35s;KV 窗口 **27392**(AutoWarmup 262→214 页,`paged KV pool: 214 context pages + 1 lookahead pages, pageLen=128, K+V=107.50 MiB`)。**对照同配方 lookup4/base 的 48384**:MTP 头 FP16(~850MB)+ lookahead 页挤占预算,窗口缩水 ~43%。

口径与 lookup 报告一致:`temperature=0, top_p=1, enable_thinking=false`,非流式,tok/s = completion_tokens / 墙钟;跑 lookup4 下翻转的 3 条(code/logic/bench_open)+ bench_echo,与 b27_base / b27_lookup4 三方对拍。

### 3.2 速度

| 条目 | base | mtp3 | 提速 | lookup4(参照) |
|---|---|---|---|---|
| code | 35.0 | **62.7** | **1.79×** | 34.3(停发) |
| logic | 35.0 | **56.9** | **1.63×** | 34.8(停发) |
| bench_echo(长文复述) | 34.3 | 69.0 | 2.01× | **92.1(2.68×)** |
| bench_open(开放生成) | 35.1 | 38.6 | 1.10× | 34.4(停发) |

### 3.3 接受率(负载不敏感)

服务端统计(时序,对应 code→logic→bench_open→bench_echo):

```
[Qwen3.5 MTP] pos_accept_rate=[93.75%, 84.38%, 76.56%].
[Qwen3.5 MTP] pos_accept_rate=[85.94%, 68.75%, 59.38%].
[Qwen3.5 MTP] pos_accept_rate=[85.42%, 70.83%, 61.46%].
[Qwen3.5 MTP] pos_accept_rate=[81.25%, 60.16%, 48.44%].
[Qwen3.5 MTP] pos_accept_rate=[78.75%, 56.25%, 41.88%].
[Qwen3.5 MTP] pos_accept_rate=[77.34%, 56.77%, 43.75%].
```

E[每步接受草稿] ≈ p1+p1p2+p1p2p3:code 段 ≈ 2.3(3.3 tok/步 → 1.79×)、末段 ≈ 1.4(2.4 tok/步 → 2.01×),与实测提速自洽。**对比 lookup4 的 [6%…68%] 负载敏感曲线**:MTP 头对内容类型不挑,普适 ~50-80%;lookup 只在"输出复述输入"时高。

### 3.4 正确性:翻转归因(本轮核心结论)

三方对拍矩阵(base / lookup4 / mtp3,同 prompt 同 greedy):

| 条目 | lk4==base | mtp3==base | lk4==mtp3 | 解读 |
|---|---|---|---|---|
| code | ✗(208/208,注释一词之差) | ✓ | ✗ | 翻转点取决于草稿上下文,有随机性 |
| logic | ✗(350/350) | ✗(350/350) | **✓ byte-exact** | 两后端翻转后内容**完全一致**、双双偏离 base |
| bench_open | ✗(425/415) | ✗(383) | ✗ | 开放生成近并列密集,三方各异 |
| bench_echo | ✓ | ✓ | ✓ | 复述负载无近并列,三方 byte-exact(517 tok) |

**结论**:logic 条目 lookup4 与 MTP3 翻转后 byte-exact 互证——翻转源是两条路径**共享的多 token 验证 forward**(全位置 logits、batch 改变 kernel/累加序、fp4 KV)与逐步 forward 的数值差异,与草稿来自查表还是 MTP 头无关。greedy 语义无损(logic 两版均为等价数学表述;math 各配置答案一致)。该结论同时为 MTP 与 lookup 的正确性口径定调:**"语义无损 + 近并列翻转可解释"**。

## 4. MTP vs prompt-lookup(dense 27B 实测对照)

| | MTP3 | lookup4 |
|---|---|---|
| 草稿来源 | 训练过的 nextn 头 forward | prompt/已生成 token 的 n-gram 查表 |
| 额外成本 | MTP 头 FP16 ~850MB + 每步草稿 forward + 1 lookahead 页 | 零(无草稿模型、免训练、零显存) |
| 同配方 KV 窗口 | 27392 | 48384 |
| 复述型(bench_echo) | 69.0(2.01×) | **92.1(2.68×)** |
| 通用负载(code/logic) | **56.9-62.7(1.63-1.79×)** | ≈1×(自适应停发) |
| 开放生成 | 1.10× | 0.98×(停发) |
| 接受率特征 | 负载不敏感([77-93%, 57-84%, 42-77%]) | 负载敏感(6%→68%) |
| 采样模式 | greedy exact + 采样 typical | 仅 greedy exact(采样回退普通解码) |
| 模型要求 | 需自带 nextn 头(且过 HasMtpWeights 融合检查) | 任意 qwen3_5 族 |

**选型**:负载可预判为翻译/改写/RAG 摘要/长文复写 → lookup(更快且零成本);通用推理/对话 → MTP;`--mtp` 与 `--lookup` 互斥(组合启动报错)。

## 5. MoE 上 MTP 现状(未实测,边界说明)

- Ornith-1.5-35B(qwen35moe)GGUF 实含 `blk.40.nextn.*` 头(本轮 `strings` 确认),但部署配方即 `--mtp 0`,本轮矩阵(base/lookup4/lookup8)未纳入 MTP 对照。
- 构造分析(非实测):`mtpVerifyGraphEligible` 要求 `num_experts == 0` → **MoE 上 MTP 验证步同样全 eager**,叠加 GGUF MoE 专家 kernel 零批摊销,与 lookup 相同的回归机理预期同样成立;此为 lookup 验证中根因链的直接推论,待 MoE 验证图支持落地后一并复测。
- Flash-Next FP8 在 3090 上 MTP warmup 直接崩(issue #717),与量化方案无关。

## 6. 结论清单

1. **MTP 在 dense Q4 GGUF 上成熟可用**(ForceFP16 remap 补丁后):通用负载 1.6-1.8×、复述 2.0×、开放 1.1×,接受率普适且负载不敏感。
2. **速度波动看接受率,不看引擎**:块节拍稳定,tok/s = E ÷ 节拍;测速必须固定 prompt + 采样参数。
3. **正确性口径与投机解码共性**:近并列翻转为共享验证 forward 的数值性质(MTP 与 lookup 在 logic 条目 byte-exact 互证);greedy 语义无损。
4. **代价**:~850MB MTP 头 + lookahead 页 → 同配方 KV 窗口 48384→27392(-43%);显存紧张时是真实取舍。
5. **与 lookup 互补**:复述负载 lookup 胜且零成本;通用负载 MTP 胜;MoE 上二者均受验证图 dense-only 限制(lookup 已实测回归,MTP 为分析推论)。

## 7. 工件与复现

本轮工件:`/tmp/lookupval/`(已同步副本 `/ssd1/syncthing/data/Prompt-Lookup-验证-2026-09-12/`):

- `b27_mtp3.json` — MTP3 四条结果;`b27_base.json` / `b27_lookup4.json` — 对照
- `s27b_mtp3.log` — 启动 + 接受率统计原文;`start27b.sh` — 启动脚本(mtp3 模式)
- `battery.py` / `compare.py` — 负载与对拍脚本

复现:`start27b.sh mtp3` → `FT_MODEL=qwen38_27b_q4 python3 battery.py out.json --bench` → 三方对拍(需先停生产 8896 释放显存)。历史部分详见 fastllm 仓库 local-patches 分支 commit(PR #721 / issue #717)与本机部署脚本 `/ssd1/scripts/start_qwen38_27b_q4.sh`。
