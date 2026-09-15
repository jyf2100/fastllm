# dsv41-shared-attention

## ADDED Requirements

### Requirement: 跨层共享的压缩 KV 生产与消费
系统 SHALL 仅在 kv_source 层(由 `kv_source_layer_ids` 指定,默认 [2,8,14,20])运行 Compressor 并写入该请求的共享压缩 KV 缓存;其余 `compress_ratio > 0` 的层 MUST 直接读取共享缓存,不得自行计算压缩 KV。共享缓存按请求隔离,层间依赖靠顺序执行保证(源层先于消费者执行),无需加锁。

#### Scenario: 非源层读共享缓存
- **WHEN** 前向经过 L21-39(ratio 1,非 kv_source)
- **THEN** 这些层读取 L20 产出的共享压缩 KV,不调用 Compressor,前向结果与参考实现一致

#### Scenario: 多请求隔离
- **WHEN** batch 内多个请求并发 decode
- **THEN** 每个请求的共享压缩 KV 互相隔离,无跨请求读写

### Requirement: Compressor ratio 1
ratio-1 层的 Compressor SHALL 为无门控纯投影:`wkv → norm`,每 token 产出一行 512 维 latent(无跨步状态、无组聚合);checkpoint 中该层无 `wgate` 属预期。

#### Scenario: ratio 1 每 token 一行
- **WHEN** ratio-1 层处理长度为 L 的输入
- **THEN** 压缩缓存得到恰好 L 行 latent,第 i 行仅由第 i 个 token 经 wkv+norm 产出

### Requirement: Compressor ratio 2 的门控聚合与跨步状态
ratio-2 层的 Compressor SHALL 以 softmax 门控聚合相邻 2 个 token 为一个 latent,并跨 decode 步维护 fp32 的 kv_state/score_state 部分组状态;组未满时 MUST NOT 产出 latent(返回空),组满才产出。latent MUST 先以 pre-RoPE 形式交给 Indexer,再施加 RoPE(组首位置 = 组索引 × 2)并量化写入共享压缩 KV 缓存。

#### Scenario: 奇数 token 的部分组跨步
- **WHEN** 输入累积 token 数为奇数(存在未配对 token)
- **THEN** 该 token 的状态保留在 fp32 ring 中,下一个 token 到达后合并产出 latent;压缩缓存行数 = ⌊total/2⌋

#### Scenario: latent 先 pre-RoPE 后入缓存
- **WHEN** ratio-2 组边界产出 latent
- **THEN** Indexer 使用该 latent 的 pre-RoPE 版本打分;压缩 KV 缓存写入的是 RoPE 后(仅尾部 64 维 rope_head_dim)版本

### Requirement: 共享 Indexer 与两级候选 top-k
系统 SHALL 仅在 index_source 层运行 Indexer:`q/k` 打分、因果 mask、top-`index_topk`(512)选块;`candidate_source_layer`(L20)SHALL 先运行 `select_candidate_blocks`(从候选缓存 top-`candidate_topk_blocks`(2048)个块,块大小 `candidate_block_size`(8))并发布候选集合,其后的 index_source 层 MUST 仅在候选集合内打分选块;非 index_source 层直接复用最近 index_source 层发布的 top-k 结果。

#### Scenario: 两级约束生效
- **WHEN** L24/L28/L32/L36 的 Indexer 选块
- **THEN** 选块范围被 L20 发布的候选集合约束,且 top-k 结果按位置排序、不可达位置标记为 -1

#### Scenario: 非源层复用 top-k
- **WHEN** L21-23(非 index_source)注意力前向
- **THEN** 其压缩侧注意力索引复用 L20 发布的 topk_idxs,不运行 Indexer 计算

### Requirement: 滑窗注意力与注意力核心
每层 SHALL 维护 128 token 滑窗 KV ring(单头 512 维 latent,post-RoPE 后 FP8 量化存储);注意力输出 SHALL 为 `[滑窗 KV; 共享压缩 KV]` 拼接上的稀疏注意力,使用 per-head `attn_sink` 与 softmax_scale;输出经 `wo_a`(o_groups=8 分组低秩,einsum 语义)+`wo_b` 投影,并对输出尾部 rope 维施加逆 RoPE。

#### Scenario: 滑窗 ring 滚动
- **WHEN** decode 超过 128 token
- **THEN** ring 按位置取模覆写,每步注意力窗口覆盖最近 128 个原始 token

#### Scenario: 拼接稀疏注意力
- **WHEN** ratio>0 层执行注意力
- **THEN** 参与计算的 KV = 滑窗 128 行 + 共享压缩 KV 中 top-k 选中的行,输出与参考实现数值一致(容差内)

### Requirement: YaRN 仅作用于压缩分支相关 RoPE
RoPE SHALL 按 `rope_theta`/`compress_rope_theta` 与 yarn(factor 16)配置计算;主分支与压缩分支使用各自的频率表,行为对齐参考实现。

#### Scenario: 长上下文外推
- **WHEN** 上下文超过原始训练长度(65K)进入 yarn 区间
- **THEN** RoPE 频率按 yarn 缩放,输出不发散(与参考实现对拍一致)

### Requirement: 共享缓存的前缀缓存与图捕获适配
前缀缓存 Record/Query SHALL 以请求为单位整体记录 4 组共享压缩 KV(组不可拆分);CUDA graph 捕获(启用时)SHALL 为共享张量使用固定地址缓冲;两者关闭时系统 MUST 仍可正确推理。

#### Scenario: 前缀缓存整体记录
- **WHEN** 请求完成并记录前缀缓存,随后同前缀新请求查询
- **THEN** 4 组共享压缩 KV 一并命中或一并未命中,不出现部分命中导致的错位

#### Scenario: 无 CUDA graph 正确性
- **WHEN** 关闭 CUDA graph 运行 decode
- **THEN** 输出与开启时一致(性能差异不影响正确性)
