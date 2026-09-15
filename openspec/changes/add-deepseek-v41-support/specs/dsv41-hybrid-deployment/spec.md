# dsv41-hybrid-deployment

## ADDED Requirements

### Requirement: CPU-hybrid 设备放置配方
系统 SHALL 支持以下设备放置并给出启动配方:注意力/共享专家/embed/head/GKV 于单卡 GPU;路由专家(MXFP4)与 Engram 表驻留 NUMA host 内存;专家计算走 NUMA MoE 管线。GPU 常驻集(不含 KV)SHALL ≤ 16GB(24G 卡可部署)。

#### Scenario: 单卡 24G 部署
- **WHEN** 在 24G GPU + ≥700GB RAM 机器按配方启动
- **THEN** 模型完成加载,显存占用 ≤ 16GB(不含 KV 池),进入服务状态

#### Scenario: 专家 NUMA 流水
- **WHEN** prefill/decode 经过 MoE 层
- **THEN** 路由专家自 NUMA 内存流式计算,不 OOM、不阻塞 GPU 常驻权重

### Requirement: KV 预算符合共享架构
系统文档与实测 SHALL 验证 KV 占用:滑窗 ring ≈ 2.6MB/请求封顶(40 层 × 128 × 512B);共享压缩 KV ≈ 1.25KB/token(fp8);45K 上下文单请求 KV < 100MB。

#### Scenario: 长上下文 KV 实测
- **WHEN** 单请求累积 45K token 并测量全部 KV 结构占用
- **THEN** 总量 < 100MB,与按公式的预算一致(±20%)

### Requirement: 性能验收基准
在 ≥700GB RAM、8 通道 DDR4 EPYC 级 CPU + 单 GPU 机器上,系统 SHALL 达到:decode ≥ 10 tok/s(1 并发)、prefill ≥ 200 tok/s(1K+ prompt);结果 MUST 以固定 bench 脚本产出并记录(参照 Qwen3.8-Flash numa 配方的测试方法)。

#### Scenario: decode 基准
- **WHEN** 运行固定 decode bench(1 并发,中等上下文)
- **THEN** 吞吐 ≥ 10 tok/s 且输出正确性已由对拍保障

#### Scenario: prefill 基准
- **WHEN** 运行固定 prefill bench(2K-5K token prompt)
- **THEN** 吞吐 ≥ 200 tok/s

### Requirement: 端到端正确性对拍
系统 SHALL 提供 greedy 输出对拍:固定 prompt 集的输出与 HF 参考实现(`inference/model.py`)逐 token 一致;并 SHALL 与 vLLM(`vllm/vllm-openai:deepseekv41-flash-0909` 或更新)做交叉抽检。

#### Scenario: 与 HF 参考一致
- **WHEN** 固定 prompt 集(≥5 条,含短/中/长与代码/中文)greedy 推理
- **THEN** 全部与 HF 参考实现输出逐 token 一致

#### Scenario: 与 vLLM 交叉抽检
- **WHEN** 任选 2 条 prompt 对比 vLLM 输出
- **THEN** 一致或差异可解释(采样参数差异)

### Requirement: 低内存机器可验证
在 RAM 小于权重总量的机器上(如 503GB),系统 SHALL 支持 mmap 懒加载模式完成正确性验证:短 prompt 推理可用,RSS 随命中页增长且不 OOM。

#### Scenario: 503GB 机器冒烟
- **WHEN** 以 mmap 模式加载并在短 prompt 上推理
- **THEN** 完成推理且 RSS 峰值 < 机器物理内存的 90%
