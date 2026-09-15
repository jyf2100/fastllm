# Spec Delta: prompt-lookup-drafting

## ADDED Requirements

### Requirement: 引擎为纯 token 空间的独立组件
n-gram 草稿引擎 SHALL 以独立组件形式实现,其公开接口仅接受/返回 token 序列(`std::vector<int>`),MUST NOT 依赖任何模型代码、权重或设备资源。相同组件 SHALL 可被不同模型族的推测 harness 复用而无须修改。

#### Scenario: 无模型环境下单元测试
- **WHEN** 在不链接任何模型实现的测试目标中,以合成 token 序列调用引擎的构建与查询接口
- **THEN** 引擎编译、链接、运行均成功,行为仅由输入 token 序列决定

### Requirement: 从 token 序列构建 n-gram 索引
给定 token 序列与窗口范围 [n_min, n_max],引擎 SHALL 为序列中每个可行位置、每个 n ∈ [n_min, n_max] 建立索引条目:key 为该位置前 n 个 token,value SHALL 支持取得该位置起的**连续续写切片**(至多草稿上限个 token)。构建 SHALL 从序列尾部反向进行,容量封顶时丢弃头部古旧条目、保留近因条目。

#### Scenario: prompt 含重复片段(末现覆盖)
- **WHEN** 序列 `A B C D A B C E` 以窗口 [1,4] 构建索引,随后以 `A B C` 为最近 token 查询
- **THEN** 引擎返回基于**最近一次** `A B C` 出现位置的续写(即 `E`),而非首次出现位置的 `D`

#### Scenario: 序列过短
- **WHEN** 序列长度 ≤ n_min(无法形成任何 key)
- **THEN** 索引为空,任何查询返回空草稿,不报错

#### Scenario: 容量封顶保留近因
- **WHEN** 构建超长序列使条目数达到上限
- **THEN** 索引条目数不再增长,序列尾部的近因条目仍可命中,头部古旧条目被丢弃,进程不崩溃、内存占用有界

### Requirement: 重复 key 的消解策略
同一 key 在序列中多次出现时,引擎 SHALL 取该 key **最近一次出现位置**的续写(recency/末现覆盖);增量追加新 token 后的索引语义 MUST 与用完整序列重建完全一致。实现 MUST NOT 让容器迭代顺序渗入候选选择。

#### Scenario: 增量追加等价于全量重建
- **WHEN** 同一完整序列分别以"全量构建"与"分段构建+增量追加"两种方式建立索引,对同一最近 token 查询
- **THEN** 两种方式返回的草稿完全相同

### Requirement: 增量更新
引擎 SHALL 支持以追加方式把新提交的 token 并入既有索引,追加源 MUST 限定为已发布/已提交 token(不含因终止条件中断而未发布的后继 token)。

#### Scenario: 生成侧自我重复同样可加速
- **WHEN** prompt 无重复,但已生成部分包含重复短语,新 token 追加进索引后查询最近 token
- **THEN** 命中生成侧建立的条目并返回草稿

### Requirement: 最长匹配优先的连续续写拷贝
查询时引擎 SHALL 从 n_max 降序探测到 n_min,取首个命中;命中后 SHALL 拷贝该位置起的连续续写切片,长度至多草稿上限 N;续写不足 N 时 SHALL 以已产出续写为后缀继续探测延伸;延伸中途所有 n 均无命中时 SHALL 返回**当前部分链**(不是空)。草稿链长度 MUST NOT 超过 N(1..8)。

#### Scenario: 长匹配优先于短匹配
- **WHEN** 最近 token 同时命中 4-gram 与 1-gram 条目,且两者续写不同
- **THEN** 草稿取 4-gram 命中位置的连续续写切片

#### Scenario: 链长截断
- **WHEN** 命中位置的连续续写长度超过草稿上限 N
- **THEN** 返回的草稿长度恰为 N,且为该位置起的前 N 个连续 token

#### Scenario: 延伸中途断链返回部分链
- **WHEN** 首次命中产出部分续写,以该续写为后缀再探测时所有 n 均无命中
- **THEN** 返回首段命中的部分草稿链,而非空列表

#### Scenario: 无任何命中
- **WHEN** 所有 n ∈ [n_min, n_max] 的 key 均未命中
- **THEN** 返回空草稿列表

### Requirement: 确定性
在索引内容与最近 token 相同的前提下,引擎的查询结果 SHALL 是确定的(相同输入产生相同输出),以便验证与测试可复现。

#### Scenario: 重复查询一致性
- **WHEN** 同一索引状态下对同一最近 token 连续查询两次
- **THEN** 两次返回的草稿完全相同

### Requirement: 容量与生命周期边界
索引条目数 SHALL 受可配置上限约束(env `FASTLLM_LOOKUP_MAX_ENTRIES`,默认 262144),超限时 MUST 以有界方式拒绝新增而非无界增长;索引生命周期 SHALL 绑定单个请求,请求结束后随上下文释放,MUST NOT 跨请求泄漏条目。

#### Scenario: 生命周期绑定请求
- **WHEN** 一个请求结束、其上下文被释放后,新请求开始
- **THEN** 新请求的索引不含前一请求的任何条目
