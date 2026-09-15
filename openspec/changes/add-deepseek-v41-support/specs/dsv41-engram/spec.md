# dsv41-engram

## ADDED Requirements

### Requirement: token 压缩映射 bit-exact 构建
系统 SHALL 从 tokenizer 构建 token id → 压缩 id 映射(归一化键:NFKC→NFD→StripAccents→Lowercase→连续空白折叠为单空格→首尾 Strip,纯空格 token 以私有哨兵字符保形),去重后连续编号;加载时 MUST 校验压缩词表大小等于 `engram_compressed_vocab_size`(99092),不一致 MUST 报错终止(所有哈希乘子由它派生,错位即静默劣化)。

#### Scenario: 映射大小校验通过
- **WHEN** 用官方 tokenizer 构建压缩映射
- **THEN** 得到恰好 99092 个压缩 id,加载继续

#### Scenario: 校验失败拒绝启动
- **WHEN** tokenizer 与 config 不匹配导致压缩词表大小 ≠ 99092
- **THEN** 启动报错并说明两处数值,不得进入推理

### Requirement: hash 乘子与素数桶 bit-exact 复刻
系统 SHALL 按 `np.random.default_rng(10007 × layer_id)` 的 PCG64 流生成每层 `max_ngram_size` 个 int64 乘子(界内均匀采样,×2+1 保奇),并以确定性 Miller-Rabin 从 `engram_vocab_size - 1` 起按序为每个 (n-gram 长度, head) 抽取互不重复的素数(每层 24 个),桶偏移取前缀和;C++ 实现 MUST 与 numpy 参考生成器逐 bit 一致。

#### Scenario: PCG64 对拍
- **WHEN** 运行哈希单测,以离线 numpy 生成的期望值对拍 C++ 乘子
- **THEN** 全部逐 bit 相等

#### Scenario: 素数桶布局对拍
- **WHEN** 对拍 C++ 与参考实现的每层 24 个素数及偏移
- **THEN** 完全一致,且各桶区间互不相交、总行数等于 `engram_num_embeddings`

### Requirement: n-gram 滚动哈希
每个位置 SHALL 计算 ≤4-gram 的滚动 XOR 哈希:`rolling ^= value × multiplier` 逐步累积,每个 (n-gram 长度, head) 输出 `rolling % prime + offset` 的 int32 行号;位置越界、图像 span(DEAD)或回看不足的项 MUST 以 pad 语义参与并标记阻断。回看 token MUST 取自该请求完整 token 历史(fastllm 持有全量历史,无需跨 chunk 状态)。

#### Scenario: 行号落在各自桶内
- **WHEN** 任意 token 位置计算 24 个行号
- **THEN** 每个行号严格落在对应 head 的素数桶区间内

#### Scenario: 回看取自请求历史
- **WHEN** chunked prefill 跨块边界且位置 p 需要块前 1-3 个 token
- **THEN** 直接从请求 allTokens 历史取值,哈希结果与整段一次性计算一致

### Requirement: Engram 查表与门控前向
Engram(L1/L14)SHALL 以 24 个行号 gather FP8 行(每行 256d)并按 ue8m0(每 32)scale 反量化(`scale 字节 << 23` bitcast fp32)为 bf16,经 `wkv` 线性产出每 hc 副本一个 key 与共享 value;门控 = 归一化点积(hidden 与 key 各自 RMS 归一化,per (token, hc) 归一化而非联合)经 signed-sqrt 后 sigmoid,输出 = `hidden + gate × value` 写回 residual。表驻留 host(见 dsv41-model-loading),GPU 路径 SHALL 经 UVA 按行 gather。

#### Scenario: 门控数值对拍
- **WHEN** 对拍参考实现 Engram 前向(相同输入与权重)
- **THEN** 输出在 bf16 容差内一致

#### Scenario: 文本路径图像位掩码
- **WHEN** 输入含被标记为 DEAD 的位置(图像 span)
- **THEN** 这些位置门强制为 0,residual 原样通过

### Requirement: Engram 不进入 GPU 常驻
Engram 模块 MUST NOT 将表拷贝到 GPU 常驻显存;查表激活(24×256d)与 wkv 计算可按设备配置执行。

#### Scenario: 显存占用不随表增长
- **WHEN** 启用与禁用 Engram 查表对比 GPU 常驻显存
- **THEN** 差值仅来自激活缓冲(MB 级),无表拷贝
