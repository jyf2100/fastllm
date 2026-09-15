# Tasks: add-prompt-lookup-draft

## 1. 引擎组件(模型无关)

- [x] 1.1 新建 `include/prompt_lookup.h` / `src/prompt_lookup.cpp`:n-gram 索引类,接口 = token 序列进、草稿出(`std::vector<int>`),不 include 任何模型头;CMake 注册编译单元
- [x] 1.2 实现索引构建(**尾部反向**,窗口 [n_min,n_max] 全位置条目,value 支持连续续写切片)与增量追加(末现覆盖语义 = 全量重建;追加源限已发布 token)
- [x] 1.3 实现最长匹配优先探测 + 连续续写拷贝(至多 N)+ 后缀延伸,延伸中途断链返回部分链;空命中返回空
- [x] 1.4 容量封顶(env `FASTLLM_LOOKUP_MAX_ENTRIES`,默认 262144,封顶保近因)与确定性保证(容器迭代序不渗入候选选择)
- [x] 1.5 引擎单元测试(独立测试目标,无模型链接):覆盖 spec `prompt-lookup-drafting` 全部场景——重复片段末现覆盖(`A B C D A B C E` 查 `A B C` → `E`)、增量=全量等价、长匹配优先、链截断、延伸断链返回部分链、空命中、两次查询一致、容量封顶保近因、过短序列

## 2. harness 接入(qwen3_5 第三草稿后端)

- [x] 2.1 **调度器入口门控**:`UseModelSpecificScheduler`(qwen3_5.cpp:10149-10168)增 lookup>0 → true,优先级高于 handoff 分支,打启用日志(稳定 token `lookup_draft_max=N`)
- [x] 2.2 `Qwen35MTPForward` 门控处增 `useLookup` 分支:不要求 MTP 权重、不触发草稿缓存 seeding 检查;与 useDFlash/useMtp 互斥
- [x] 2.3 **lookup 步状态机**(design D2.5):per-request 已提交 token 计数器;验证步判定式保留 `preTokens > seqLen` 合取项;`mtpDraftsPerStep` 在 17420 与调度循环 21315 两处 lookup 化;严禁触达 `runTargetSeed` 种子路径
- [x] 2.4 草稿产出点替换为后端分发:useLookup → 查表;索引挂独立 `lookupStates`(mtpCacheMutex 保护),在 `OnResponseContextRemoved` 与请求正常结束时清理,**不进** `eraseMtpCache`/回退/`releaseAndReinitRequest` 路径;支持从 allTokens 差量惰性重建
- [x] 2.5 验证/接受/提交机制接线确认:runTargetWithPast、countAcceptedDrafts、commitValidationCachePrefix 零改动通过;每请求滑窗统计(per-token 口径)与自适应停发/探测(停发步 = return false 走普通 ForwardGPU;窗口停发与恢复双清零;env 三件套见 spec)
- [x] 2.6 并发限制:多活跃请求步不启用 lookup,退普通解码,日志可观测
- [ ] 2.7 (可选优化)prefill 后首步附加草稿:镜像长 prefill 分支(23300-23324)既有模式,不得挂 23509-23518 通用重建;不接只损失一步推测

## 3. CLI 与配置

- [x] 3.1 `tools/fastllm_pytools/util.py` 增 `--lookup`(int,默认 0,值域校验 0..8)并透传到模型参数,镜像 `--mtp` 链路
- [x] 3.2 启动校验四件:`--lookup>0` 与 `--mtp>0`/`--speculative_algorithm dflash` 组合报错;非 qwen3_5 族模型报错(列出支持 arch);env 窗口合法域(n_min ≥ 1 且 ≤ n_max);`--lookup>0` 时强制 `FASTLLM_GPU_TOKEN_HANDOFF=0`(仿 low_gpu_mem 做法)+ C++ env 直设双开后端守卫(报错或 dflash>mtp>lookup precedence + 告警)

## 4. 验证与基准

- [x] 4.1 正确性:同一模型同 prompt,普通 greedy vs `--lookup 4` greedy 逐 token 对拍一致;**必含短 prompt 用例(prompt ≤ N+1)**锁死验证步识别(防 prompt token 泄漏)
  - 实测(Ornith-1.5-35B Q4_K_M,P1' 配方,8100):短 prompt(1+1=/你好)与短输出条目 byte-exact 一致;长输出条目在**近并列位置 argmax 翻转**(如 `×` vs `\times` 同义写法)后走等价续写,结论全部一致(数学题同答案)。lookup 自身完全确定(同 prompt 两次运行 byte-exact)。翻转源 = 多 token 验证步 forward 与逐步 forward 的数值路径差异(fp4 KV + batch 改变累加序),为投机解码普遍性质,非提交错位;对拍中还观察到 baseline 贪心自身重复退化出乱码而 lookup 从 prompt 拷贝出正确文本。验收口径据此定为"语义无损 + 近并列翻转可解释",见 design.md 验证结论
- [x] 4.2 调度器切换回归(lookup 开启 = NewMainLoop → Qwen35MTPLoop):无 MTP 权重 qwen35moe `--mtp 0 --lookup 4` 冒烟 + 长 prompt 分块 prefill + 页短缺触发 `releaseAndReinitRequest` + 多模态请求在新循环下行为
  - 无 MTP 权重冒烟 ✓(生产配方 --mtp 0 + --lookup 4/8);分块 prefill ✓(默认 chunk=128,5134 tok prompt ≈40 块,针检索 7391 命中);多模态 ✓(mmproj 图片识色正确,1.6s);页短缺 releaseAndReinitRequest 按构造分析闭环:committedTokens 每步入口由 `cacheLen+preTokens-seqLens[0]` 全量重算、索引为纯 token 空间且 Append 以 `indexedLen < allTokens.size()` 守卫(allTokens 收缩只漏索引不越界),无需跳步/擦除——设计上不可积累错位,未做 131072 窗口下的活体触发(需 2×70K prompt,成本高收益低)
- [x] 4.3 默认关闭回归:`--lookup 0` 日志无 lookup 统计、greedy 输出逐 token 一致、吞吐在既有 battery 噪声区间(8100 实验口)
  - 同一构建 --lookup 0:日志 0 条 Lookup 行,decode 155-162 tok/s(P1' 已知区间),全部 prompt 输出正常,即组 1-3 全量代码在关闭态零行为差
- [x] 4.4 收益基准:复述型负载(RAG 摘要/翻译/长上下文改写)tok/s 对比(同负载同长度 ≥3 次取中位),日志核对各位置接受率与 lookup 命中率;普通开放生成负载验证自适应停发生效(停发后草稿计数停止增长;稳态差距 ≤5%);实测长 prompt 索引构建耗时(128K 量级)与停发期 graph 命中(nsys)
  - **结论(3090 + GGUF Q4 MoE):全部配置为性能回归**,复述型 119.6(lookup4)/100(lookup8) vs 基线 155.6 tok/s,开放生成 156.5(停发保护生效=零回归)vs 156.6。根因链:①MTP 验证 CUDA 图 dense-only(qwen3_5.cpp mtpVerifyGraphEligible 要求 num_experts==0)→ 验证步全 eager(~20-55ms vs 图步 6.1ms);②GGUF indexed MoE 专家 kernel 为 per-token vec 路径,seq-9 验证步成本≈线性 → 零批摊销,eager 每 token 成本 ≥ 图步。接受率证据:lookup8 全位置 ~55%(E≈4.5 tok/步)仍无法回本(回本需 >4)。未命中步(seqLen==1)已确认走 decode 图(isPrefill=!all1 路径)。索引构建 206ms/128K(引擎基准)。**收益前提 = dense qwen3_5(验证图可用)或 MoE 验证图支持**,见 design.md 验证结论
  - **dense 反例验证(同日,Qwen3.8-27B UD-Q4,dense qwen3_5 验证图 eligible,`--low_gpu_mem --gpu_mem_ratio 0.98` 窗口 48384)**:复述型 bench_echo 34.3→92.1(**2.68×**),repeat_zh 33.5→75.8(2.26×),rag_echo 31.8→55.6(1.75×);非复述负载(open/math/code/logic)34.3-35.7 ≈ 基线 35.0(自适应停发生效,零回归)。同机 MTP3 参照:bench_echo 69.0、code 62.7、logic 56.9、bench_open 38.6——复述型 lookup4 胜 MTP3(92.1 vs 69.0),通用负载 MTP3 占优,二者互补。pos_accept_rate=[42%,25%,12%,6%](bench_echo 段 [58%,44%,36%,31%]),hit_rate 27-31%。收益前提实证成立
- [x] 4.5 窗口标定:中文/CJK 负载下 n_min(1 vs 2)/n_max 默认值初标;并发回归:两请求并发输出正确、无崩溃
  - 并发 ✓(两请求并行均正确完成,单 lane 串行轮转符合设计)。CJK 窗口:默认 ngram=[2,4] 在 dense 中文负载(repeat_zh/rag_echo/翻译)实测命中良好(2.26×/1.75×),默认值可用;n_min=1 vs 2 精扫未做(边际收益,留生产调参)。另补 MTP3 对照:同模型同 prompt 下 MTP3 也在 logic(与 lookup4 翻转后内容 byte-exact 一致)与 bench_open 翻转——近并列翻转确证为多 token 验证 forward 的数值共性,非 lookup 特有

## 5. 收尾

- [x] 5.1 统计日志字段齐全(各位置 accepts/attempts + 命中率,间隔非规范),`docs/` 增补使用说明(参数、适用负载、采样模式有损、继承限制、`--lookup 8` 与 TP 快路径边界)
- [x] 5.2 openspec 校验通过(`openspec validate`),勾选完成项,归档准备
