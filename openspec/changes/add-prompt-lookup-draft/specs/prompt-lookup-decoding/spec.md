# Spec Delta: prompt-lookup-decoding

## ADDED Requirements

### Requirement: CLI 开关与默认关闭
系统 SHALL 提供 `--lookup N` 参数(整数 0..8,0 为默认)控制 prompt-lookup 推测解码的每步草稿上限;`--lookup 0` 时系统 SHALL 与不带本特性的基线无行为差异,且日志 MUST NOT 出现任何 lookup 统计输出。

#### Scenario: 默认关闭零差异
- **WHEN** 以不带 `--lookup` 的参数启动服务并完成推理
- **THEN** 日志无 lookup 统计输出,greedy 输出与改动前基线逐 token 一致,吞吐落在既有 battery 的噪声区间内

#### Scenario: 启用
- **WHEN** 以 `--lookup 4` 启动
- **THEN** 服务正常启动,日志含稳定标记 `lookup_draft_max=4`

#### Scenario: 参数越界
- **WHEN** 以 `--lookup 9` 或非整数 `--lookup abc` 启动
- **THEN** 启动报错退出并说明合法域 0..8

### Requirement: 调度器入口激活
lookup 为唯一推测后端(无 MTP 权重、无 DFlash、`--mtp 0`)时,系统 SHALL 使模型进入模型专用调度循环(`Qwen35MTPLoop`)而非通用循环,推测解码 MUST 实际生效(多 token 提交统计出现);激活判定 MUST NOT 静默失败。

#### Scenario: 纯目标模型启用
- **WHEN** 加载一个不含 MTP 权重的 qwen35moe 模型并以 `--lookup 4` 启动,发送复述型请求
- **THEN** 日志出现 `lookup_draft_max=4` 与多 token 提交统计(存在接受草稿数 ≥1 的验证步),而非仅单 token 推进

### Requirement: 验证步识别
系统 SHALL 依据 per-request 已提交 token 计数与输入构成判定"多 token 输入 = 待验证草稿",判定条件 MUST 区分 prompt 尾部与草稿:prompt 长度 ≤ 草稿上限+1 的请求 MUST NOT 将 prompt token 当作草稿验证或提交。

#### Scenario: 短 prompt 不泄漏
- **WHEN** prompt 仅 2 个 token,以 `--lookup 4` greedy 生成
- **THEN** 输出 token 序列与普通 greedy 解码完全一致,prompt token 不出现在输出中

### Requirement: 草稿后端与 handoff 互斥
`--lookup >0` 与 `--mtp >0` 或 `--speculative_algorithm dflash` 同时给出时,系统 SHALL 在启动时报错退出;`--lookup >0` 时系统 SHALL 确保 GPU token handoff 不激活(含 auto fast path 自动开启的场景与 C++ env 直设途径);同一调度步 MUST NOT 激活多个草稿后端。

#### Scenario: 与 MTP 冲突
- **WHEN** 以 `--lookup 4 --mtp 2` 启动
- **THEN** 启动失败并给出明确的互斥错误信息

#### Scenario: handoff 被抑制
- **WHEN** 以满足 auto fast path 条件的参数加 `--lookup 4` 启动(该组合原本会自动开启 GPU token handoff)
- **THEN** GPU token handoff 保持关闭,日志可确认推测分支在调度循环中生效

### Requirement: 适用域
`--lookup >0` 加载非 qwen3_5 族(qwen3_5/qwen35moe/qwen3_5_moe 之外)模型时,系统 SHALL 在启动时报错退出并说明当前支持的 arch 列表,MUST NOT 静默忽略。

#### Scenario: 不支持的 arch
- **WHEN** 加载 llama/glm 系模型并以 `--lookup 4` 启动
- **THEN** 启动失败并给出支持域说明(而非静默空转)

### Requirement: 窗口配置
n-gram 窗口 SHALL 经 env 配置:`FASTLLM_LOOKUP_NGRAM_MIN`(默认 2)与 `FASTLLM_LOOKUP_NGRAM_MAX`(默认 4);非法配置(n_min < 1 或 n_min > n_max)MUST 在启动时报错退出;默认值在负载标定后 SHALL 可调整而无须改代码。

#### Scenario: 窗口合法生效
- **WHEN** 以 `FASTLLM_LOOKUP_NGRAM_MIN=3 FASTLLM_LOOKUP_NGRAM_MAX=6` 启动
- **THEN** 服务正常启动,查询行为符合 3..6 窗口语义

#### Scenario: 窗口非法
- **WHEN** 以 `FASTLLM_LOOKUP_NGRAM_MIN=5 FASTLLM_LOOKUP_NGRAM_MAX=3` 启动
- **THEN** 启动报错退出

### Requirement: 验证与提交语义与 MTP 等价
prompt-lookup 模式 SHALL 通过既有多 token 验证前向(`speculativeCollectAllLogits`)与既有接受判据产生提交:greedy(温度 0)用逐位 argmax 精确比对,采样模式用 typical 判据(与 MTP 相同阈值);接受前缀的 KV 页与线性状态提交、拒绝部分的回滚 SHALL 复用既有机制。greedy 无损的适用前提与 MTP 相同:带 repeat_penalty/output_logits/tool-call 约束的请求走普通解码路径,不在本模式范围内。

#### Scenario: greedy 无损
- **WHEN** 同一模型同一 prompt(含短 prompt 用例)分别以普通 greedy 解码与 `--lookup N` greedy 解码生成完整回答
- **THEN** 两次输出 token 序列完全一致

#### Scenario: 接受的草稿整块发布
- **WHEN** 某验证步接受了 k 个草稿
- **THEN** 该步向流式输出发布 k+1 个 token(锚点 + 接受草稿),与 MTP 的块发布语义一致

#### Scenario: 无草稿命中时的行为
- **WHEN** 某验证步查表无命中(空草稿)
- **THEN** 该步输入仅含锚点 token,输出推进 1 个 token,等同普通解码步

### Requirement: 自适应降级与恢复
系统 SHALL 按请求维护滑动窗口接受率,口径 MUST 为 per-token:窗口内接受草稿数 ÷ 发出草稿数。窗口 W(env `FASTLLM_LOOKUP_PAUSE_WINDOW`,默认 64)内接受率低于阈值(env `FASTLLM_LOOKUP_PAUSE_THRESHOLD`,默认 0.10)时 MUST 停发草稿,停发步 SHALL 走普通单 token 前向路径(与普通解码同构);停发期间每 R 个解码步(env `FASTLLM_LOOKUP_PROBE_INTERVAL`,默认 256)SHALL 发一次满长度探测草稿,命中即恢复。滑动窗口 SHALL 在进入停发时与恢复时清零重建,恢复后 MUST NOT 因停发前历史立即再次停发。

#### Scenario: 非复述负载自动停发
- **WHEN** 负载为普通开放生成,草稿持续不命中
- **THEN** 系统在窗口期内停发草稿(停发后草稿统计计数停止增长),稳态吞吐与普通解码的差距 ≤5%(测量协议:8100 battery,同负载同长度 ≥3 次取中位)

#### Scenario: 停发后负载转为复述型恢复
- **WHEN** 停发后生成进入引用上下文的阶段,探测草稿被接受
- **THEN** 草稿发送恢复,滑动窗口自恢复点重建,后续失败不与停发前历史合并

#### Scenario: 恢复不震荡
- **WHEN** 恢复后的首个探测草稿被拒绝
- **THEN** 系统不立即再次停发(窗口已清零重建,需重新积累 W 步证据)

### Requirement: 并发限制(MVP)
当调度步存在多个活跃请求时,该步 SHALL 不启用 prompt-lookup(退普通解码);单活跃请求步 SHALL 正常启用。此限制 SHALL 在日志中可观测。

#### Scenario: 并发时安全回退
- **WHEN** 两个请求同时活跃于同一调度器
- **THEN** 两个请求均以普通单 token 步推进,输出正确,无崩溃

### Requirement: 统计与可观测
系统 SHALL 输出含以下字段的统计:各位置 accepts/attempts(沿用 MTP 口径)与 lookup 命中率;统计 SHALL 周期性打印,打印间隔为非规范实现细节;格式无须逐字节复刻 MTP 日志,但字段 MUST 齐全可解析。

#### Scenario: 日志可核对收益
- **WHEN** 一次复述型请求完成
- **THEN** 日志含各位置接受率与 lookup 命中率字段,可据此计算每步平均提交 token 数
