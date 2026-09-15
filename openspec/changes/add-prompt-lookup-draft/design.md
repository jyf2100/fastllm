# Design: add-prompt-lookup-draft

## Context

fastllm 的推测解码基础设施集中在 qwen3_5 harness(`Qwen35MTPLoop` 调度循环,`src/models/qwen3_5.cpp:21304`),被 `qwen3_5`/`qwen35moe`/`qwen3_5_moe` 三个 arch 共用(生产 Ornith = qwen35moe)。该 harness 的每步流程为:

```
[调度器] nextInputTokens(anchor + drafts) → 构张量
   → [验证] runTargetWithPast: 目标模型一次前向 1+N token(speculativeCollectAllLogits=true)
   → [接受] countAcceptedDrafts: greedy=逐位 argmax 精确比对 / 采样=typical 判据(与 MTP 相同)
   → [提交] commitValidationCachePrefix: KV 页 + 线性状态 slot 按接受前缀提交,拒绝部分回滚
   → [草稿] runSpeculativeDraftChain: MTP 草稿层前向(或 DFlash 后端)产出下一批 drafts
   → [发布] acceptedTokenLists → resultTokenQueue + dictCV.notify_all
```

源码注释(18386 附近)明示设计意图:"verifier 共享,草稿生成器保持独立后端";`runSpeculativeDraftChain`(18468-18482)是全部 4 处草稿产出点(18502/19523/19772/19832)的唯一汇聚分发点。验证/接受/提交机制与草稿来源完全解耦——这是本设计的立足点:**n-gram 查表是第三个草稿后端,连草稿前向都不需要**。

参考实现:llama.cpp `--lookup-*`(lookup-decode.cpp / lookup-merge.cpp)、PLD 论文(Saxena, 2023)。核心结论:对输出复述输入的负载可达 2-4× 提速;草稿 = 查 prompt n-gram 表的续写,零草稿模型成本。

## Goals / Non-Goals

**Goals:**
- 模型无关的 n-gram 草稿引擎,作为可复用的独立组件(纯 token 空间接口)
- 接入 qwen3_5 harness 成为第三草稿后端,与 MTP/DFlash/handoff 互斥,复用全部验证/接受/提交机制
- 无 MTP/DFlash 权重的模型(生产 Ornith 现状)可直接启用——含调度器入口门控
- 普通负载(草稿不命中)无可感知回归:自适应停发草稿
- CLI 语义对齐 `--mtp`:`--lookup N`(0=关闭,1..8 每步草稿上限)

**Non-Goals:**
- basellm 通用调度循环泛化(llama/glm/qwen4exp 等非 qwen3_5 族接入)——引擎已为此解耦,接入是后续独立变更;本期对非 qwen3_5 族 + `--lookup>0` 直接启动报错
- 采样模式下的无损化(rejection-sampling 判据)——与 MTP 现状对齐,保持 typical
- 跨请求静态草稿缓存(prompt-hash 共享索引)——与前缀缓存协同是后续课题
- 多请求 batched 草稿(每 lane 独立 drafts)——MVP 沿用 MTP 单 lane 策略

## Decisions

### D1: 引擎独立成组件,不内嵌 qwen3_5.cpp
新文件(如 `include/prompt_lookup.h` + `src/prompt_lookup.cpp`),接口只有 `std::vector<int>` 进出,不 include 任何模型头。备选:直接写在 qwen3_5.cpp 里(改动更小)——否决,因为用户明确要求通用性,且二期 basellm 泛化时内嵌版等于重写。

### D2: 首个接入点 = qwen3_5 harness 第三后端,不做 basellm 通用循环
备选:直接改 `RunNewMainLoop`(所有模型共用的生产最热路径,需跨模型族泛化 KV 回滚)——爆炸半径大。接入点共 **四处**(前三处复用既有机制,第四处是激活链路的必要条件):

- **(a) 调度器入口门控 `UseModelSpecificScheduler`(qwen3_5.cpp:10149-10168)**:lookup>0 时返回 true 进入 `Qwen35MTPLoop`,**优先级高于 handoff 分支**,并打启用日志。不接这处,lookup-only 配置(无 MTP/DFlash 权重,如生产 Ornith)根本进不了专用循环——特性静默死亡。
- (b) `Qwen35MTPForward` 门控处(17316-17325)增 `useLookup` 分支:不要求 MTP 权重、不触发草稿缓存 seeding 检查;与 useDFlash/useMtp 互斥
- (c) 草稿产出点(19508 一带)`runSpeculativeDraftChain` 调用替换为后端分发:useLookup → 查表
- (d) 调度器 prefill 后首步输入构造处(镜像长 prefill 分支 23300-23324 给 DFlash/MTP 附加首草稿的既有模式)同样附加查表草稿——**可选优化**,不接只损失一步推测;不得挂到 23509-23518 的通用重建(会污染多请求步)

### D2.5: lookup 步状态机(验证步识别)——承重设计,不可临场发挥
现有 harness 用 `activeCacheTokens` 判别"多 token 输入 = 待验证草稿"(`isSpeculativeValidation`,18486-18488),并参与 17376/17380/17384-17408 多处门控;`mtpDraftsPerStep` 在 **17420 与调度循环 21315 两处**决定草稿数。lookup 没有草稿 KV 缓存,必须显式定义等价物:

- **per-request 已提交 token 计数器**(挂独立 lookup 状态,见 D3 生命周期),判定式形如"输入含草稿 ⇔ preTokens > seqLen 且 seqLen ≤ drafts+1"——**必须保留 `preTokens > seqLen` 合取项**,否则 prompt 长度 ≤ N+1 的短 prompt 会被误判为验证步,prompt token 被当草稿"验证"提交、泄漏进输出
- `mtpDraftsPerStep` 两处均须 lookup 化(取 `--lookup` 值),漏一处即每步 fallback
- **严禁误入种子路径** `runTargetSeed`(18264-18301):它用真实 pastKeyValues 跑 ForwardGPU、全部输入 token 的 KV 无条件写缓存、盲提交 1 token——lookup 步不得触达

### D3: 索引 = 每请求动态 n-gram 哈希表,首解码步惰性构建 + 增量更新
- key = 最近 n 个 token(n ∈ [n_min, n_max]),**value = 源序列位置**;续写 = 该位置起的**连续切片**(PLD 式整段拷贝,至多 N 个),不做逐 token 再探测的马尔可夫游走。备选:value=单 token + 链式再探测——否决,n_min 较小时等价低接受率的 1-gram 游走,PLD 采用连续拷贝即为此因。
- **重复 key 消解 = 末现覆盖(recency)**:同一 key 多次出现时,取**最近一次出现位置**的续写。理由:长上下文/多轮对话里最近的重复才是相关续写,"先插入者胜"(emplace)会系统性偏向 prompt 头部;且"增量追加语义 = 全量重建"正是靠末现覆盖达成。实现不得让哈希表迭代序渗入选择。备选:llama.cpp 式计数 boost——列为标定期的可选增强。
- 延伸:命中续写不足 N 时,以已产出续写为后缀再探测延伸;**中途全 n 无命中 → 返回当前部分链**(不是空)。
- 追加源 = **已发布 token**(allTokens 差量);提交循环在 EOS 处 break、`acceptedTokenLists` 可能含未发布后继,不得作追加源(避免污染索引质量)。
- **容量与构建顺序**:env `FASTLLM_LOOKUP_MAX_ENTRIES`(默认 262144,内存账 ~100B/条 ≈ 25MB/请求)。构建从序列**尾部反向进行**,封顶时丢弃的是头部古旧区,**近因条目必存活**——正向构建会让 128K prompt 的后半段(续写价值最高)全部被砍。备选:按 n 分预算/LRU——复杂度不值。
- **生命周期**:独立 map(`lookupStates`),挂 `mtpCacheMutex`,在 `OnResponseContextRemoved`(10140-10147,fetch 线程经 basellm.cpp:579-585 调用)与请求正常结束时清理;**不得**进 `eraseMtpCache`(每次中途 fallback 都擦,23360-23362)与 `releaseAndReinitRequest` 路径(21401-21414)——否则 128K 上下文每次回退重建付数十毫秒、停发窗口计数清零重学。清理后可从 allTokens 差量惰性重建容错。
- 探测:从 n_max 降到 n_min 取首个命中;构建成本 O(len×n_max),128K prompt ~0.5M 次哈希插入为**数十毫秒**量级(实现期实测标定),摊销进解码。

### D4: 开关与互斥
- `--lookup N`(util.py,镜像 `--mtp` 的参数风格与透传链路);窗口 env `FASTLLM_LOOKUP_NGRAM_MIN`(默认 2)/`FASTLLM_LOOKUP_NGRAM_MAX`(默认 4)。n_min 默认 2:1-gram 兜底草稿近噪声(标定开放问题见下)。
- 互斥(一个调度步只允许一个草稿后端 + handoff):
  - CLI:`--lookup>0` 与 `--mtp>0` 或 `--speculative_algorithm dflash` 组合 → 启动报错
  - **handoff:`--lookup>0` 时 util.py 强制 `FASTLLM_GPU_TOKEN_HANDOFF=0`**(仿 low_gpu_mem 在 580 行的做法)。原因:`--lookup 4`(mtp=0、非 dflash、CUDA)恰好命中 auto fast path 的 eligible 条件(util.py:572-576)→ handoff 被自动开启,其消费分支(22844)排在推测分支(23339)之前且可自链续占——不强制关掉,停发后的探测草稿会被饿死,首次停发即永久死亡
  - C++ env 直设守卫:`FASTLLM_QWEN35_ENABLE_MTP` 与 lookup env 同时 >0 → 启动报错或明确 precedence(dflash > mtp > lookup)+ 告警日志,防运维绕过 CLI 校验
- `--lookup 8` 边界:seqLen=9 > `QWEN35_MTP_FAST_SEQ_MAX=8`(qwen3_5.cpp:470),TP-inplace 与 batch 快路径被排除、走通用重放——正确但慢,TP 配置下文档建议有效值 ≤7。

### D5: 接受判据与提交机制原样复用
greedy → exact(无损,输出与普通 greedy 解码逐 token 一致);采样 → typical(与 MTP 相同的 posterior_threshold=0.09 / alpha=0.30,有损但分布有界)。`commitValidationCachePrefix` 的 KV 页 + 线性状态 slot 提交对草稿来源不敏感,不改。**终止语义无需新增护栏**(经代码核实):提交循环逐 token 检查 EOS/stop(23590-23608)与 output_token_limit(23620),一步提交 k+1 token 不可能越界;查表草稿的 token id 天然合法(来源 allTokens)。继承限制(与 MTP 相同):repeat_penalty/output_logits/tool-call 约束的请求走普通解码路径。

### D6: 自适应降级(普通负载防回归)
- **口径**:per-token 接受率 = 窗口内接受草稿数 ÷ 发出草稿数(与统计日志"按位置联合概率"口径区分,显式定义)。
- 滑动窗口 W(env `FASTLLM_LOOKUP_PAUSE_WINDOW`,默认 64):低于阈值(env `FASTLLM_LOOKUP_PAUSE_THRESHOLD`,默认 0.10)→ **停发草稿**;停发步的实现 = `Qwen35MTPForward` 返回 false → 落到 23483 普通 ForwardGPU(单 token,CUDA graph 资格路径不变)。备选:带空草稿走种子路径——否决,种子路径强制捕获隐藏态跑 eager 前向,graph 不可用,5% 回归承诺悬空。
- **窗口重置**:进入停发时清零;恢复(探测草稿被接受)时也清零重建——否则恢复后首探被拒会立即再停发,陷入 2-R 步震荡。
- 探测:每 R 个**解码步**(env `FASTLLM_LOOKUP_PROBE_INTERVAL`,默认 256)发一次**满 N 长度**探测草稿,命中即恢复。
- 备选:llama.cpp 式动态缩窗(`--lookup-dynamic`)——列为 Open Question。

### D7: 并发策略 MVP 沿用 MTP 单 lane
多活跃请求的调度步不启用 lookup(该步退普通解码),与 MTP 在单 GPU 上的 lane 策略一致。机制上 `nextInputTokenLists` 本就是 per-request 向量,batched 草稿是结构性可行的后续增强,不在本期。线程安全:索引与计数器只在唯一调度线程(前向线程)touch;跨线程的上下文删除走 D3 的 mutex + 钩子模式。

### D8: 常量与上限
每步草稿上限沿用 `QWEN35_MTP_MAX_DRAFTS=8`(`--lookup` 取值 0..8);统计复用 MTP 的 accepts/attempts 计数器(`QWEN35_MTP_LOG_INTERVAL` 打印间隔**非规范**,字段级要求见 spec),另加 lookup 命中率计数;启用日志含稳定 token `lookup_draft_max=N`。

## Risks / Trade-offs

- [采样模式有损] typical 判据下输出分布有偏 → 与 MTP 现状一致,greedy 无损;文档明示。
- [无命中负载的稳态开销] 每 N 步多算 N 个位置的 logits 行(权重流量不变,激活/softmax 略增) → D6 自适应停发(停发步走 return false,graph 路径不变);默认关闭,显式开启才承担。
- [**调度器整体切换的爆炸半径**] 开 lookup 即从 `NewMainLoop` 切到 `Qwen35MTPLoop`——分页预算/准入(21600-21803)、chunked prefill、驱逐路径全然不同 → tasks 4.2/4.5 增加分块 prefill、页短缺 `releaseAndReinitRequest`、多模态请求在新循环下的显式回归场景。
- [短 prompt 误判为验证步] prompt ≤ N+1 时判定式接错会泄漏 prompt token 进输出 → D2.5 状态机保留 `preTokens > seqLen` 合取项;tasks 4.1 短 prompt 对拍用例锁死。
- [CPU 每步探测开销] O(n_max × 链长) 哈希查询,微秒级 → 相对 40-50ms 验证步可忽略。
- [索引内存/构建延迟] 每请求上限条目封顶(~25MB/请求 @ 默认)、尾部反向构建保近因;长 prompt 首解码步多付数十毫秒构建 → 惰性构建 + 首步即可多 token 提交,净收益为正;实测标定。
- [流式中断竞争] 一步发布 k+1 token 与客户端 abort → 与 MTP 现状相同的既有语义,不新增风险面。
- [默认关闭] `--lookup 0` 零差异 → 所有现有部署无变化。

## Migration Plan

1. 合入后默认 `--lookup 0`,所有现有部署零变化。
2. 实验实例(8100 口)先开 `--lookup 4` 用 RAG/翻译负载验证;greedy 逐 token 对拍普通解码(含短 prompt 用例)确认无损。
3. 生产启用 = 重启加参数;回滚 = 重启去掉参数。无状态迁移、无模型文件变化。

## Open Questions

- 停发期 CUDA graph 是否实际命中(理论上 return false 走普通前向即恢复资格,需 nsys 实测确认)。
- 动态缩窗(llama.cpp `--lookup-dynamic` 式)是否优于"停发+周期探测"?
- 中文/CJK 负载的最优窗口(n_min 1 vs 2、n_max 默认值)需实测标定。
- 重复 key 的 recency 覆盖 vs 计数 boost(llama.cpp 式)在长上下文多轮对话下的接受率对比。
- 跨请求静态索引与前缀缓存(prompt-hash)协同的收益边界。

## 验证结论(2026-09-12,roc-epyc 3090,Ornith-1.5-35B Q4_K_M P1' 配方)

> 完整验证报告(含两模型 base/lookup/MTP 对拍明细与复现步骤):`docs/prompt-lookup-validation.md`

运行时验证(8100 实验口,沙箱本地构建)对上面两个开放问题给出了实测答案:

### 正确性:语义无损成立,bit 级对拍需重新定义

- 短 prompt(≤N+1)与短输出条目 byte-exact 一致;长输出条目在**近并列位置 argmax 翻转**(如 `×` vs `\times`)后走等价续写,结论一致(数学题同答案)。
- lookup 自身完全确定:同 prompt 同配置两次运行 byte-exact——排除提交错位/状态泄漏类 bug。
- 翻转根因 = 多 token 验证 forward(全位置 logits)与逐步 forward 的数值路径差异(fp4 KV、batch 改变 kernel/累加序),是投机解码的普遍性质(MTP/DFlash 同理),**非 lookup 引入**。对拍中还观察到 baseline 贪心在复述场景自身退化出乱码、lookup 反而从 prompt 拷贝出正确文本的案例。
- 结论:验收口径从"逐 token bit 级一致"修正为"**语义无损 + 翻转仅发生于近并列 logits 且可解释**"。(备选后续:借用 DFlash exact-verify 的 `FastllmCudaSetLinearExactBatchThreshold` 机制探索 bit 级一致,代价未评估。)

### 性能:MoE GGUF 上为回归,收益前提是验证图

实测(基线 155.6 tok/s):复述型 119.6(lookup4)/100.5(lookup8);开放生成 156.5(自适应停发生效,零回归)。接受率不是瓶颈(lookup8 全位置 ~55%,E≈4.5 tok/步),根因链:

1. **MTP 验证 CUDA 图 dense-only**:`mtpVerifyGraphEligible` 要求 `num_experts == 0`(qwen3_5.cpp ~16272),MoE 验证步全 eager(~20-55ms/步 vs 图步 6.1ms);
2. **GGUF indexed MoE 专家 kernel 是 per-token vec 路径**:seq-9 验证步成本≈线性增长,零批摊销 → eager 每 token 成本 ≥ 图步,回本需 E>4 仍不够。
3. 未命中步(seqLen==1)已走 decode 图(`isPrefill=!all1` 路径确认),无额外税。

**适用边界**:dense qwen3_5(验证图 eligible)或未来 MoE 验证图支持之前,`--lookup` 在 GGUF Q4 MoE + 单卡部署上是性能负资产;停发保护保证最坏情况≈基线(开放生成实测零回归)。停发期 CUDA graph 命中已由吞吐等价证实(无需 nsys)。

### dense 反例(同日,Qwen3.8-27B UD-Q4,验证图 eligible):收益前提实证成立

同机同构建,`--lookup 4`(`--low_gpu_mem --gpu_mem_ratio 0.98`,窗口 48384),基线 34.3-35.1 tok/s:

| 负载 | base | lookup4 | 提速 | MTP3 参照 |
|---|---|---|---|---|
| bench_echo(长复述) | 34.3 | **92.1** | **2.68×** | 69.0 |
| repeat_zh | 33.5 | 75.8 | 2.26× | — |
| rag_echo | 31.8 | 55.6 | 1.75× | — |
| code / logic / math / open | 34.3-35.7 | 34.3-35.7 | ≈1×(停发) | 56.9-62.7 |

- **正确性**:11 条 battery 8 条 byte-exact;3 条 DIFF 中 code(208/208)与 logic(350/350)token 数不变 = 单点近并列翻转,bench_open 425/415。**MTP3 对照决定性**:logic 条目 MTP3 与 lookup4 翻转后内容 byte-exact 一致、双双偏离 base;bench_open MTP3 也翻转(383 tok)——翻转确证为多 token 验证 forward 数值共性,与草稿来源无关。
- **与 MTP 互补**:复述型 lookup4 胜 MTP3(92.1 vs 69.0,且免草稿模型/免训练/零显存);通用负载 MTP3 占优(其接受率普适 ~50%)。停发保护使 lookup 在低命中负载始终 ≈ 基线。
- pos_accept_rate bench_echo 段 [58%,44%,36%,31%],hit_rate 27-31%;默认 ngram=[2,4] 在 dense 中文负载直接可用。
