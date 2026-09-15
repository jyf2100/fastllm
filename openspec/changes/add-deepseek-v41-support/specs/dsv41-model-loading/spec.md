# dsv41-model-loading

## ADDED Requirements

### Requirement: 注册并识别 deepseek_v41 模型类型
fastllm SHALL 在模型工厂中注册 `model_type=deepseek_v41` / `deepseek_v41_text` 与架构名 `DeepseekV41ForCausalLM`,创建 `DeepSeekV41Model` 实例;现有 `deepseek_v4` 注册路径 MUST NOT 受影响。

#### Scenario: 加载 deepseek_v41 配置
- **WHEN** 输入模型的 config.json 含 `"model_type": "deepseek_v41"`(或其 text 变体)且权重为 HF safetensors 格式
- **THEN** fastllm 实例化 DeepSeekV41Model 并完成全部权重加载,不报"unsupported model type"

#### Scenario: deepseek_v4 不回归
- **WHEN** 加载现有 DeepSeek-V4-Flash(deepseek_v4)权重
- **THEN** 仍走 DeepSeekV4Model 路径,行为与本变更前一致

### Requirement: 解析 V4.1 专属配置字段
DeepSeekV41Model SHALL 解析以下字段并在缺失关键项时报错:`compress_ratios`(40 项数组)、`kv_source_layer_ids`、`index_source_layer_ids`、`candidate_source_layer_id`、`candidate_topk_blocks`、`candidate_block_size`、`index_topk`、`engram_layer_ids`、`engram_num_embeddings`、`engram_max_ngram_size`、`engram_n_heads`、`engram_head_dim`、`engram_compressed_vocab_size`、`engram_vocab_size`、`engram_pad_token_id`、`sliding_window`、`hc_mult`、`o_groups`、`qk_rope_head_dim`、`expert_dtype`。

#### Scenario: 关键字段缺失
- **WHEN** config.json 缺少 `kv_source_layer_ids` 或 `compress_ratios` 长度与层数不符
- **THEN** 加载失败并给出明确错误信息,指出缺失/不合法字段

### Requirement: 按真实张量命名映射权重
加载器 SHALL 按官方 checkpoint 命名(`layers.N.*`、`mtp.Z.*`、`embed.*`、`head.*`、`vision.*`,无 `model.` 前缀)映射权重,并满足真实 index 的出现次数约束:compressor 权重仅存在于 kv_source 层(ratio-1 层无 `wgate`)、indexer 的 `wk/k_norm` 仅存在于 kv_source 层、`wq_b/weights_proj` 仅存在于 index_source 层、engram 权重仅存在于 `engram_layer_ids` 层、`mtp.Z` 的 `main_proj/main_norm` 仅 stage 0、`markov_head/confidence_head/norm` 仅末 stage。

#### Scenario: 张量全集加载
- **WHEN** 加载官方 48-shard checkpoint
- **THEN** 全部 96,085 张量按名映射成功,无"未识别张量"警告遗漏主干权重

#### Scenario: 非源层缺失 compressor/indexer 权重属预期
- **WHEN** 某非 kv_source 层不存在 `layers.N.attn.compressor.*` 张量
- **THEN** 加载器不将其视为缺失错误,该层按共享读取方运行

### Requirement: 量化 dtype 布局
加载器 SHALL 按以下布局加载:linear/注意力权重(`wkv/wq_a/wq_b/wo_a/wo_b/shared_experts/engram.wkv/main_proj`)为 FP8 E4M3 + 32×32 block ue8m0 scale(`.scale` 张量);路由专家(`ffn.experts.E.w1/w2/w3`)按 `expert_dtype` 加载,fp4 时为 MXFP4(block 32,ue8m0);`embed/head/gate.*/q_norm/kv_norm/attn_sink/hc_*` 为 bf16/fp32 原样。

#### Scenario: MXFP4 专家加载
- **WHEN** `expert_dtype="fp4"`(DeepSeek-V4.1-Flash)
- **THEN** 专家权重以 NVFP4_BLOCK_32_E8M0(或等价布局)加载,单专家显存占用 ≈ 18.8MB(35.4M 参数 × 0.5B + 32 分组 ue8m0 scale)

#### Scenario: gate 双 bias
- **WHEN** 加载 `layers.N.ffn.gate.bias` 与 `layers.N.ffn.gate.bias_vl`
- **THEN** 两者均保留;文本推理路径仅使用 `gate.bias`

### Requirement: Engram 大表 host 驻留
Engram 嵌入表(`engram.embed.weight/.scale`,各 ~101GB)SHALL 驻留 host 内存且 MUST NOT 计入 GPU 显存预算(通过 `IsHostOnlyWeight` 类钩子排除),SHALL 支持按行惰性读取;表 MUST 保持 FP8+ue8m0 原始布局,不做加载期反量化。

#### Scenario: GPU 预算不含 Engram 表
- **WHEN** 以 `--moe_device cuda` 或 spill 配置启动且 GPU 预算规划运行
- **THEN** Engram 两表(~203GB)不参与 GPU 层放置计算,budget 计算与无 Engram 时一致

#### Scenario: 大表 mmap 懒加载
- **WHEN** 以 mmap/低内存模式加载(机器 RAM 小于权重总量)
- **THEN** 模型可完成加载并推理,实际 RSS 只随路由命中的页增长

### Requirement: mtp 权重可选加载
加载器 SHALL 支持跳过 `mtp.*` 权重(DSpark 未实现阶段),并 SHALL 不因跳过而影响主干权重加载与推理正确性。

#### Scenario: 跳过 mtp
- **WHEN** 以配置/env 指定跳过 `mtp.*` 张量启动
- **THEN** 主干 40 层推理正常,mtp 权重不占内存
