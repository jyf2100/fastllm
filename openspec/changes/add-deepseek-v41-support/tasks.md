# Tasks: add-deepseek-v41-support

## 1. P0 摸底与基建(动工前提)

- [ ] 1.1 确认 fastllm safetensors reader 的 mmap/懒加载能力(读代码得出结论,不行则列出需要 mmap 的大张量类型清单)
- [ ] 1.2 确认性能验收机器(aidev 系列 RAM ≥700GB?)与权重分发方式(48 shards ~510GB 从 hf-mirror 拉取的落盘计划)
- [ ] 1.3 下载官方 tokenizer 文件,离线生成三份期望值:压缩 token 映射(129280→99092)、每层 PCG64 乘子、每层 24 素数及桶偏移(存 /tmp 供单测对拍)
- [ ] 1.4 建立 HF 参考实现对拍脚手架:用官方 model.py 跑固定 prompt 集,落盘逐 token 输出与关键层 hidden-state 快照

## 2. 模型骨架与加载

- [ ] 2.1 拷贝 deepseekv4 骨架创建 `include/models/deepseekv41.h` + `src/models/deepseekv41.cpp`,注册工厂(model_type deepseek_v41/deepseek_v41_text),先空跑加载失败路径
- [ ] 2.2 实现全部 V4.1 config 字段解析与校验(缺失/长度不符报错)
- [ ] 2.3 按真实张量命名实现权重映射(attn/compressor×4/indexer×8/engram×2/mtp/gate.bias_vl),含"非源层无 compressor/indexer 权重属预期"逻辑
- [ ] 2.4 接通量化布局:FP8 32×32 ue8m0 linear、MXFP4 专家(NVFP4_BLOCK_32_E8M0)、bf16 embed/head/gate;单测:单专家加载后 ~18.8MB、抽样反量化值与 safetensors 原值对拍
- [ ] 2.5 实现 `IsHostOnlyWeight` 排除 Engram 两表 + mtp 跳过加载开关
- [ ] 2.6 若 1.1 结论为需要:给超阈值 host-only 张量加 mmap 懒加载路径;在 503GB 本机冒烟(短 prompt RSS 记录)

## 3. 共享注意力运行时

- [ ] 3.1 定义 `DeepSeekV41SharedAttnState`(per-request:compressKV×4、indexK×4、topkIdxs×8、candidates、candidateBlocks),挂请求上下文,接入 prefill/decode 生命周期
- [ ] 3.2 滑窗 ring + 拼接稀疏注意力 + attn_sink + wo_a 分组低秩 + 逆 RoPE:从 V4 迁移并适配 head_dim 512 / window 128(先 CPU 参考路径)
- [ ] 3.3 Compressor ratio 1(无门控纯投影,每 token 一行直写)
- [ ] 3.4 Compressor ratio 2(fp32 kv_state/score_state 跨步 ring、组满产出、pre-RoPE 交 indexer → RoPE+量化入缓存)
- [ ] 3.5 Indexer:打分、因果 mask、top-512 选块(排序、不可达 -1);先 bf16 打分保证正确性
- [ ] 3.6 两级候选:L20 select_candidate_blocks(top-2048 块 × 8)写 candidates,后续 index_source 在候选内 mask;非 index_source 层复用 top-k
- [ ] 3.7 YaRN 主/压缩双分支频率表;65K+ 长度冒烟不发散
- [ ] 3.8 模块级对拍:L1-L3 + L20 邻域逐层 hidden-state vs HF 参考(容差内)

## 4. Engram

- [ ] 4.1 C++ 实现 PCG64(与 numpy default_rng 逐 bit)+ Miller-Rabin 素数桶 + 桶偏移;单测对拍 1.3 期望值
- [ ] 4.2 加载期构建压缩 token 映射(归一化键去重),校验 99092(不符即 ErrorInFastLLM)
- [ ] 4.3 n-gram 滚动哈希前向(24 行号/token/层,回看取 allTokens,pad/DEAD 语义)
- [ ] 4.4 查表 + ue8m0 反量化(`<<23` bitcast)+ wkv + 门控(per-hc RMS、signed-sqrt、sigmoid)写回 residual;CPU 路径
- [ ] 4.5 Engram 数值对拍:与 HF 参考 Engram 前向 bf16 容差一致;显存差值测试(表不进 GPU)

## 5. 端到端正确性(P1 收口)

- [ ] 5.1 40 层全模型 CPU/CPU-hybrid 前向打通(greedy)
- [ ] 5.2 固定 prompt 集逐 token 对拍 HF 参考(≥5 条:短/中/长/代码/中文);修到全绿
- [ ] 5.3 与 vLLM(docker deepseekv41-flash-0909)交叉抽检 2 条
- [ ] 5.4 前缀缓存整体记录/查询(4 组共享 KV 原子性)开关两态正确性;无 CUDA graph 模式正确性

## 6. 性能路径(P2)

- [ ] 6.1 CUDA kernel 化:compressor ratio1/2、engram lookup(UVA gather,持久 grid)+ gate、两级 top-k、滑窗/稀疏注意力 CUDA 版(V4 内核迁移)
- [ ] 6.2 MXFP4 专家接 NUMA MoE 流水线(套 Qwen3.8-Flash numa 配方:FT_NUMAS 绑定、spill 预算)
- [ ] 6.3 CUDA graph 捕获:共享张量固定地址缓冲,graph 开/关正确性与性能对比
- [ ] 6.4 (可选)FP4 压缩 KV(288B/token,抄 vLLM #56344 布局)
- [ ] 6.5 部署文档:启动配方、KV 预算表(2.6MB ring + 1.25KB/token 实测)、bench 结果(decode ≥10 tok/s、prefill ≥200 tok/s 于验收机)

## 7. 回归与交付

- [ ] 7.1 deepseek_v4 / qwen3_5 / qwen4exp 现有部署冒烟无回归(新注册项纯增量)
- [ ] 7.2 openspec 归档:specs 合入 openspec/specs/,变更关闭
