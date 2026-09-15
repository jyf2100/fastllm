# Proposal: add-prompt-lookup-draft

## Why

fastllm 目前的推测解码只有两种草稿来源,都要求"草稿器本身要花成本":MTP 需要模型自带 MTP 权重并每步跑草稿层前向,DFlash2 需要 3.6GB 外挂草稿模型。两者对"输出复述输入"的负载(翻译、改写、RAG 摘要、代码补全、多轮对话中的上下文引用)都有先天盲区——这类负载里最优草稿就藏在 prompt 已有 token 里,查表即可,零草稿成本、免训练、不占显存。

现有 qwen3_5 推测 harness 的验证/接受/提交机制(`Qwen35MTPForward`)是草稿来源无关的,源码注释(18386)明示"草稿生成器保持独立后端"。补一个 n-gram 查表草稿后端是工程量最小的推测算法增量,且生产 Ornith(qwen35moe,当前 `--mtp 0` 运行)无需任何 MTP 权重即可受益。

## What Changes

- 新增**模型无关的 n-gram 草稿引擎**(独立组件,不依赖任何模型代码):对请求的 prompt+已生成 token 建立动态 n-gram 索引,解码时以最近 token 为 key 查表产出草稿候选
- 在 qwen3_5 推测 harness(qwen3_5/qwen35moe/qwen3_5_moe 共用)中接入为**第三个草稿后端**,与 MTP/DFlash 互斥;复用既有多 token 验证前向、exact/typical 接受判据、线性状态提交机制
- 新增 CLI `--lookup N`(每步草稿数,0=关闭,上限 8,对齐 `--mtp` 语义)及 env 调参(窗口大小等);与 MTP/DFlash/GPU-token-handoff 互斥(含 auto fast path 自动开启场景)
- 接受率自适应降级:近期接受率过低时自动停发草稿退回单 token 步,周期性重试,保证普通对话负载无可感知回归
- MVP 范围:单请求 lane(并发>1 时该请求退回普通解码);basellm 通用调度循环泛化(llama/glm 等非 qwen3_5 族接入)列为可选后续

## Capabilities

### New Capabilities

- `prompt-lookup-drafting`: 模型无关的 n-gram 草稿引擎——索引构建/增量更新/查表续写的确定性契约、窗口与容量边界、纯 token 空间接口(为后续接入其它模型族的 harness 复用而设)
- `prompt-lookup-decoding`: 端到端推测解码模式——CLI/env 开关与互斥规则、通过既有验证-接受-提交机制产生与 MTP 等价的提交语义、无草稿命中时的行为、自适应降级与恢复、关闭时零开销

### Modified Capabilities

(无——openspec/specs/ 目前为空,本次全部为新增能力)

## Impact

- **新增**:`prompt-lookup` 引擎组件(头文件 + 实现,纯 token 空间,无模型依赖)
- **修改**:`src/models/qwen3_5.cpp` —— 调度器入口门控(`UseModelSpecificScheduler`,lookup-only 配置进 `Qwen35MTPLoop` 的必要条件)、草稿后端选择处(`Qwen35MTPForward` 的 useMtp/useDFlash 门控旁增 useLookup 分支;`runSpeculativeDraftChain` 调用点替换为后端分发)、lookup 步验证判别状态机、接受率统计复用
- **修改**:`tools/fastllm_pytools/util.py` —— 新增 `--lookup` CLI 参数并透传
- **不修改**:验证前向(`runTargetWithPast`)、接受判据(`countAcceptedDrafts`)、提交(`commitValidationCachePrefix`)、调度器 token 发布路径——全部复用
- **风险面**:默认关闭(`--lookup 0`),不影响任何现有部署;开启后新增风险集中在 qwen3_5 harness 单请求路径,与 MTP 路径共享同一套已验证机制
