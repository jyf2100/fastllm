# Design: add-deepseek-v41-support

## Context

DeepSeek-V4.1-Flash(HF `deepseek-ai/DeepSeek-V4.1-Flash`)规格(来源:真实 `model.safetensors.index.json` 96,085 张量 / 48 shards / 510.28GB,官方 `inference/model.py` 全文,vLLM `vllm/models/deepseek_v4_1/` 已合入 main):

- 40 层主干 + hc_mult=4 超连接;MoE 每层 384 路由专家(top-6)+ 1 共享,`sqrtsoftplus + noaux_tc + swiglu_limit`(与 V4 同族,fastllm 已实现)
- 注意力 = 滑窗(128)MQA-LoRA + 压缩稀疏注意力:每 token 单个 512 维共享 latent(KV),q 走 `wq_a/q_norm/wq_b`,o 走 `wo_a` 分组低秩(o_groups=8)+`wo_b`,per-head `attn_sink`
- **跨层共享**(与 V4 的本质差异):全模型仅 kv_source 层 [2,8,14,20] 运行 Compressor、仅 index_source 层 [2,8,14,20,24,28,32,36] 运行 Indexer,其余层读共享缓存;L20 额外计算 candidate blocks 约束后续 indexer(两级 top-k)
- compress_ratios:L0-1=0(纯滑窗)、L2-19=2、L20-39=1(ratio 1 = 无门控纯投影,无跨步状态)
- Engram:L1/L14,2 张 384M 行 × 256d FP8+ue8m0(每 32)表,确定性 n-gram 哈希(≤4-gram,8 hash head),门控写入 residual
- 量化:linear/attn = FP8 E4M3 32×32 block + ue8m0 scale;路由专家 = MXFP4(block 32, ue8m0);embed/head/gate/vision = bf16
- MTP 槽位(mtp.40-42)承载 DSpark(块级草稿),vision 塔为 32 层 bf16 ViT + aligner——**本次不做**
- vLLM 参考事实:Engram 表 `cpu_offload` **默认开启**(pinned host memory + UVA 读);compressor 仅实现 ratio 1/2;`tokenizer_mode=deepseek_v41`

fastllm 现状:`DeepSeekV4Model`(src/models/deepseekv4.cpp,14K 行)已实现 HC、MQA-LoRA、滑窗 ring、ratio 4/128 compressor、128-wide indexer、hash 路由 MoE、MTP+markov、FP8/FP4 专家 dtype(NVFP4_BLOCK_32_E8M0 等 4 种)、NUMA MoE 管线、投机解码验证基建。

权重账本(部署输入):路由专家 ~289GB + Engram ~203GB + DSpark 专家 ~7GB + 其余 ~11GB;GPU 侧常驻(注意力+共享专家+embed/head/vision)≈ 12-15GB;KV = 滑窗 ring 40×128×512B ≈ 2.6MB/请求(封顶)+ 压缩 KV ~1.25KB/token(fp8)→ 1M 上下文 ≈ 1.2GB/请求。

## Goals / Non-Goals

**Goals:**
- 文本模式正确推理(P1):greedy 输出与 HF 参考实现逐 token 一致
- CPU-hybrid 部署可用(P2):专家+Engram NUMA RAM 驻留、单卡 GPU 承载注意力/KV/共享专家,decode/prefill 达到 Qwen3.8-Flash numa 配方的同档水平(3090/EPYC 参照:prefill 300-500 tok/s、decode 15-25 tok/s 量级,实际以大 RAM 机器为准)
- 大 RAM 机器(≥700GB)可全量常驻;小 RAM 机器(500GB 级)可 mmap 懒加载做正确性验证

**Non-Goals:**
- vision 塔 / 多模态前向(P3,后续变更)
- DSpark 投机解码(P4,后续变更;mtp 权重允许跳过加载)
- GGUF 格式、多机 TP/EP、DSML tokenizer 前端、FP4 压缩 KV(P2 可选)
- bit-exact 复刻 vLLM 性能路径(只对齐数值正确性)

## Decisions

### D1: 新模型类独立成文件,拷贝演化而非继承 `DeepSeekV4Model`
`src/models/deepseekv41.cpp` + `include/models/deepseekv41.h`,骨架拷贝自 deepseekv4。理由:fastllm 模型层无继承惯例;V4.1 的缓存所有权重构(见 D2)会深入 V4 的 prefill/decode/CUDA-graph 路径,继承会迫使 V4 行为耦合或大量 override。代价:代码重复 ~2-3K 行,可接受(模型文件本就独立)。
*备选否决*:在 V4 内用 config 分支——会把两代架构的差异摊进所有路径,回归风险高。

### D2: 共享运行时 = 每请求 `DeepSeekV41SharedAttnState`,挂请求上下文
结构(对齐参考 `SharedAttentionRuntime` 的单槽语义):
```
struct DeepSeekV41SharedAttnState {   // per request, 4 组单槽
    Data compressKV[4];      // kv_source 层产出:2,8,14(ratio2) + 20(ratio1)
    Data indexK[4];          // 仅 owns_k 的 kv_source 层产出(供 indexer 打分)
    Data topkIdxs[8];        // 每个 index_source 层一份(候选层级不同,不共享)
    Data candidates;         // 仅 L20:candidate block 集合
    Data candidateBlocks;    // 2048 块 × block_size 8 的候选缓存
};
```
- 层间读写靠顺序执行保证("源层永远先于消费者执行"),不加锁
- **CUDA graph**:共享张量在 graph 捕获时按请求固定地址(沿用 V4 的 `cudaGraphCacheReady` 固定地址 ring 模式,扩展到跨层共享张量)
- **前缀缓存**:跨层共享使"整条前缀"必须同时包含 4 组共享缓存 → Record/Query 以请求为单位整体快照,V4 的逐层快照机制改为按共享组(4 组)记录;滑窗 KV 与共享压缩 KV 分开管理
- **投机解码/多请求**:状态与请求绑定,不全局单槽(参考实现的进程级单槽是 batch=1 的简化)

### D3: Engram 表驻留与读取
- **驻留**:mmap 只读映射 safetensors 对应 tensor 区域(表不反量化、不拷贝);`IsHostOnlyWeight()` 钩子(PL 先例)把两张表 + scale 排除出 GPU 预算
- **读取**:GPU 路径走 UVA 按行 gather(vLLM `_engram_lookup_kernel` 同构:持久 grid 限 SM 数,避开 TLB 抖动);CPU 路径直接 host gather。查表量 24 行 × 256B/token/层,带宽可忽略
- **ue8m0 反量化捷径**:`(uint8)scale << 23` bitcast fp32(ue8m0 即 2 的幂,scale 字节就是 fp32 指数域)
- *备选否决*:全表载入 GPU——197GB 不可行;加载时反量化成 bf16——内存翻倍且失去 mmap 懒加载能力

### D4: 哈希 bit-exact 三件套的 C++ 复刻 + 对拍单测
1. **token 压缩映射**(129280 → 99092):NFKC→NFD→StripAccents→Lowercase→空白折叠→Strip 的归一化键去重。实现:构建期一次性跑(bootstrap 脚本产出 `engram_token_map.bin` 随模型加载生成/缓存),运行时只查表
2. **hash 乘子**:`np.random.default_rng(10007 × layer_id)` 的 PCG64 比特流,int64 均匀采样后 ×2+1。C++ 内嵌 PCG64(算法公开,~50 行)复刻,**单测与 numpy 输出逐 bit 对拍**
3. **素数桶**:从 `engram_vocab_size - 1` 起按序抽取 24 个素数/层(Miller-Rabin),桶偏移 = 前缀和;`hashed = rolling_xor_hash % prime + offset`
- 校验门:加载时用 tokenizer 重建映射,行数 ≠ `engram_compressed_vocab_size`(99092)直接 ErrorInFastLLM(对齐 vLLM 行为)——乘子全部由它派生,错了表就整体错位

### D5: Compressor/Indexer 复用 V4 内核骨架,新增 ratio 1/2 与两级候选
- ratio 1:无 wgate/无状态,`wkv → norm` 每 token 一行,直写压缩缓存(比 ratio 4/128 简单,新写即可)
- ratio 2:复用 V4 compressor 的 fp32 `kv_state/score_state` ring 语义(已存在),组满 2 才产出 latent;latent 先出 pre-RoPE(indexer 需要),再 RoPE+量化入压缩缓存
- Indexer:q/k 走 fp4 量化打分(参考 `fp4_act_quant` block 32 e8m0;正确性阶段可用 fp8/bf16 替代,量化路径 P2);L20 的 `select_candidate_blocks`(top-2048 块)写 candidates,后续 index_source 层打分时在候选内 mask —— 单独小 kernel
- 滑窗/ring/attn_sink/sparse_attn/wo_a 分组投影:直接沿用 V4 CUDA 路径

### D6: 量化与加载复用
- 专家 MXFP4:`DataType::NVFP4_BLOCK_32_E8M0` 现成,接权重名 `layers.N.ffn.experts.E.w1/w2/w3(.scale)`
- linear FP8 32×32 ue8m0:V4 现成(wkv/wq_a/wq_b/wo_a/wo_b/shared_experts)
- bf16:embed/head/gate(+bias/bias_vl)、q_norm/kv_norm、attn_sink、hc_*、vision(P3)
- `gate.bias_vl` 文本阶段忽略(仅 image span 使用)
- **mmap 懒加载**:优先确认 fastllm safetensors reader 的 mmap 能力;若只支持全量读,给超过阈值(如 1GB)的 host-only 张量加 mmap 路径(小改动,不阻塞 P1——正确性机器 RAM 够时可全量)

### D7: 验证策略 = 三级对拍
1. 单元:哈希三件套对拍 numpy(离线生成期望值);compressor ratio 2 聚合对拍参考实现小算例
2. 模块:截取 L1-L3 + L20 邻域做逐层 hidden-state 对拍(脚本驱动 HF 参考实现)
3. 端到端:固定 prompt 集 greedy 输出 vs HF 参考实现逐 token 一致;之后与 vLLM(docker `deepseekv41-flash-0909`)对比作交叉验证

## Risks / Trade-offs

- [哈希 bit-exact 失败 → 表整体错位,输出静默劣化] → D4 校验门(行数断言)+ PCG64 对拍单测 + 首次运行打印若干哈希样例供人工比对
- [本机 503GB RAM < 510GB 权重,全量驻留不可行] → mmap 懒加载(D3/D6);正确性验证每步只触碰路由命中页,短 prompt 实测驻留 <100GB;性能验收在 ≥700GB 机器执行
- [共享缓存与 CUDA graph 捕获交互复杂(固定地址、batch 变化)] → P1 先关 CUDA graph 跑通正确性,P2 再接(沿用 V4 固定地址 ring 经验);graph 关闭时性能损失已知可接受(decode 慢 20-30%,不阻塞)
- [Engram UVA 随机读的 TLB 抖动] → vLLM 同路径已验证可行(持久 grid 限 SM);若 CPU-hybrid 配置下 GPU 侧查表慢,回退 CPU gather + H2D 小包(24×256B/token)
- [素数桶行数巨大,任何"物化全表索引"的实现都会爆内存] → gather 只按 (row, 256B) 粒度,索引即哈希值本身,无中间结构
- [tokenizer_mode deepseek_v41 的新 DSML 格式] → 文本推理用 tokenizer.json 标准路径;DSML 仅影响 chat 模板/工具调用前端,不进 P1/P2 范围
- [两级候选的 prefill 分块边界(块大小 8、topk 512)与 chunk 拼接] → 候选缓存按请求绝对位置记录,chunk 边界只影响增量计算,对齐参考实现的 offset 语义

## Migration Plan

纯新增(新 model_type 注册),不影响现有 deepseek_v4/GLM/Qwen 路径;回滚 = 删除注册项。无数据/格式迁移。

## Open Questions

- 性能验收机器的 RAM 供给(aidev 系列待查;183 为共享机且 CPU 带宽偏低)
- FP4 压缩 KV(288B/token 布局)是否值得在 P2 做(vLLM PR #56344 的布局可抄),还是 FP8 压缩 KV 已够(1M ctx 才 1.2GB)
- DSpark 阶段 mtp 权重的最小加载策略:跳过 vs 加载不用(跳过省 7GB,加载为 P4 铺路)

## 外部情报登记(2026-09-14,来源:AI4AI 028_Engram 筛选终稿 + 本仓实态核对)

> 指挥台确认 B/D 两项落地。以下登记逐项核对过仓库实态,非直接采信工单层;防重筛结论登记在项目记忆 `ai4ai-screening-register`。

### 跟踪项 1(原参考 1,改写登记):qwen4_exp 上游跟版 + 条件记忆双厂商

- **双厂商事实**:DeepSeek Engram(arXiv:2601.07372,ACL 2026)与 Qwen3.8-Flash-Next(arch=`qwen4_exp`,2026-08-26 发布 N-gram Embedding,51B 表/20M 条 {2,3}-gram/6B 激活)构成**条件记忆双厂商**——方向非单家实验,本变更的 Engram 基建(哈希三件套/host 表/门控)具备第二次复用价值,长期价值上调。
- **跟版义务**:本仓 qwen4_exp 已适配(commit `08443f04` + 14 优化提交;`src/models/qwen4_exp.cpp` 11,765 行;PLE:ngram 20M 条/128 分片/hash 乘子+桶偏移/conv 4/`--ngram_device host|disk`),跟踪点 = **上游 config/结构变更跟版**(专用代码量大,跟版成本高)。
- **checklist(并入一行,不单列跟踪)**:算子形态——RMSNorm 点积 sigmoid 门控、depthwise 因果 conv(kernel 4)——实现(qwen4_exp PLE/HyperMix)与 spec(本变更 dsv41-engram)双侧均已在仓。

### 跟踪项 2(原参考 2,价值最高):DRAM 卸载 + PCIe 预取重叠 = 验收外部锚 + 已核实的改进机会

- **外部锚**:100B 表全卸载 DRAM + 预取与计算重叠,吞吐最大惩罚 **2.8%**(abstract "<3%")。**限定语:工单层"可信(其设计内)";落地引用前必须回读 arXiv:2601.07372 核对测量口径。** 用途:本变更 `dsv41-hybrid-deployment` 吞吐验收的对标锚(197GB Engram 表 host 驻留的惩罚上限量级)。
- **仓内核查结论(2026-09-14,替代原"待查项")**:qwen4_exp PLE **无预取重叠机制**——decode/TP 路径为 pinned host + async **同流**拷贝(`qwen4_exp.cpp:3439-3466`,注释自证 "Copy and projections share a stream",异步意在避免同步阻塞与保护 host 缓冲生命周期),prefill 为 host 同步 gather+反量化。028 的"预取与前块计算重叠"对本仓是**真增量**:对症 D3 的 Engram 表读取与已知 FromCPU 流式瓶颈(qwen38fn 踩坑 §5),DSV41 P2 性能期可作候选优化项。
