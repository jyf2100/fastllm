## Why

DeepSeek-V4.1-Flash(HF `deepseek-ai/DeepSeek-V4.1-Flash`,model_type `deepseek_v41`)是 DeepSeek 最新发布的 730B 级混合专家模型(权重 510GB,48 shards),fastllm 当前只支持 `deepseek_v4`(V4-Pro/V4-Flash),无法加载该模型。该模型的形态——40 层里只有 4 层产生压缩 KV、Engram n-gram 表 197GB、专家 MXFP4 289GB——恰好落在 fastllm 独占的 CPU-hybrid 场景:注意力/共享专家/KV 放单卡 GPU(~15GB),路由专家与 Engram 表驻留大 RAM NUMA 机器,vLLM 同配置需要 8×H100 级集群。vLLM 已于 2026-09-10 将 `deepseek_v4_1` 模型树合入 main(含 Engram CPU offload 默认开启),张量命名、量化格式、哈希算法全部可考,消除了新架构接入的最大不确定性。

## What Changes

- 新增 `DeepSeekV41Model`(基于现有 `DeepSeekV4Model` 骨架演化),注册 `model_type=deepseek_v41` / `DeepseekV41ForCausalLM`,HF safetensors 直读(权重命名 `layers.N.*` / `mtp.Z.*`,无 `model.` 前缀)
- 新增**跨层共享注意力运行时**:仅 kv_source 层 [2,8,14,20] 运行 Compressor、仅 index_source 层 [2,8,14,20,24,28,32,36] 运行 Indexer,其余 30+ 层读共享缓存(替代 V4 的每层独立缓存所有权)
- 新增 **Compressor ratio 1/2**(ratio 1 无门控纯投影、ratio 2 带 fp32 ring 跨步状态;复用 V4 已有 ratio 4/128 内核骨架)
- 新增**两级候选 top-k**:candidate_source 层(20)的 select_candidate_blocks 约束后续 indexer 的选块范围
- 新增 **Engram 模块**:2 张 384M 行 FP8+ue8m0 表 host 驻留(mmap/UVA 按需读),确定性 n-gram 哈希(NFKC 归一化 token 压缩映射、per-layer numpy PCG64 乘子、素数桶布局),在 L1/L14 门控写入 residual
- 新增 MXFP4 专家路径接入(量化 dtype `NVFP4_BLOCK_32_E8M0` 等 fastllm 已支持,主要是权重映射与前向接线)
- 分期交付:文本正确性 → NUMA 性能路径 → vision 塔(bf16 简单 ViT)→ DSpark 投机解码;前两期为本次变更范围

## Capabilities

### New Capabilities
- `dsv41-model-loading`: deepseek_v41 配置解析、safetensors 张量命名映射、MXFP4/FP8-ue8m0 量化布局、Engram 大表 host 驻留加载
- `dsv41-shared-attention`: 跨层共享的 Compressor(ratio 1/2)/ Indexer / 滑窗 KV / 两级候选 top-k / 注意力前向(含 attn_sink、wo_a 分组低秩)
- `dsv41-engram`: n-gram 哈希计算(bit-exact 三件套)与 Engram 门控检索前向
- `dsv41-hybrid-deployment`: CPU-hybrid 部署配方(专家+Engram NUMA 驻留、GPU 常驻集、KV 预算)与验收基准

### Modified Capabilities
<!-- 无:deepseek_v4 现有规格不变,新模型为独立增量 -->

## Impact

- **新增代码**:`src/models/deepseekv41.cpp` + `include/models/deepseekv41.h`(骨架拷贝自 deepseekv4,预计 3-5K 行)、`src/model.cpp` 工厂注册、若干 CUDA kernel(compressor ratio1/2、engram lookup/gate、两级 top-k)
- **复用不动**:HC 超连接、MoE 路由(sqrtsoftplus/noaux_tc/swiglu_limit)、FP8 block 量化、NUMA MoE 管线、MTP 基建、YaRN——均沿用 deepseekv4.cpp 现有实现
- **部署面**:目标机需 ≥700GB RAM(权重 510GB + 运行时);本机(503GB)仅够 mmap 懒加载做正确性验证;单卡 24G GPU 即可承载注意力侧
- **参考实现**:HF 官方 `inference/model.py`(已全文精读)+ vLLM `vllm/models/deepseek_v4_1/`(engram.py/quant_config.py/compressor.py 已精读,attention.py/dspark.py 备查)
- **不依赖**:无第三方库新增;哈希复刻用 C++ 实现对齐 numpy PCG64 比特流
