# Prompt-Lookup 投机解码(--lookup)

从 prompt 与已生成 token 的 n-gram 重复片段直接查表取草稿:**零草稿模型、免训练、零额外显存**。对"输出复述输入"的负载(翻译、改写、RAG 摘要、长上下文问答、代码补全)接受率极高;普通开放生成下由自适应停发保护,实测零回归。

> ⚠️ **性能适用边界(2026-09-12,3090 实测)**:
> - **dense qwen3_5:收益实证**——Qwen3.8-27B UD-Q4 上 `--lookup 4` 复述型负载 1.75-2.68×(长文复述 34.3→92.1 tok/s,超同机 MTP3 的 69.0),非复述负载自适应停发 ≈ 基线(零回归)。
> - **MoE GGUF 单卡:全配置回归**(复述型 100-120 vs 基线 156 tok/s)——MTP 验证 CUDA 图目前 dense-only,MoE 验证步全 eager 且 GGUF 专家 kernel 无批摊销,接受率再高也回不了本;MoE 验证图支持落地前不建议开启。
> - greedy 输出语义无损,但与逐 token 解码在近并列 logits 位置可能翻转(投机解码普遍性质,MTP 同样翻转,详见 OpenSpec design.md 验证结论)。

## 使用

```bash
python -m ftllm.server <模型路径> --lookup 4        # 每步最多 4 个草稿
```

- 取值 `0..8`,默认 0(关闭);与 `--mtp` / `--dspark` / `--speculative_algorithm dflash` 互斥。
- 仅支持 qwen3_5 族(Qwen3.5 / Qwen3.5-MoE),其它架构启动报错。
- `--lookup > 0` 自动强制 `FASTLLM_GPU_TOKEN_HANDOFF=0`(handoff 消费分支会饿死 lookup 探测);CUDA graph / CUDA embedding 自动路径不受影响。
- 后端优先级:dflash > mtp > lookup;env 直设双开后端时 lookup 让位并告警一次。

## 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `FASTLLM_QWEN35_LOOKUP_DRAFTS` | 0 | 草稿数(1..8);`--lookup` 透传,一般不直设 |
| `FASTLLM_LOOKUP_NGRAM_MIN` / `_MAX` | 2 / 4 | n-gram 窗口;要求 1 ≤ min ≤ max |
| `FASTLLM_LOOKUP_MAX_ENTRIES` | 262144 | 索引容量封顶(封顶保近因,约 100B/条) |
| `FASTLLM_LOOKUP_PAUSE_WINDOW` | 64 | 自适应滑窗 token 数 |
| `FASTLLM_LOOKUP_PROBE_INTERVAL` | 256 | 停发后每多少 token 探测恢复 |
| `FASTLLM_LOOKUP_PAUSE_THRESHOLD` | 0.10 | 窗口接受率低于该值即停发 |

## 语义要点

- **key** = 位置前 n 个 token(n ∈ [n_min, n_max]);**续写** = 该位置起的连续切片(PLD 式整段拷贝);重复 key 末现覆盖;最长匹配优先,续写耗尽后以已产出草稿为后缀再探测延伸,断链返回部分链。
- **索引源只含已发布 token**(prompt + 已确认生成),永不摄入未验证草稿——被拒草稿不可能污染后续查询。
- **greedy 无损**:验证/接受/提交复用 MTP 的共享验证器,逐 token 精确对拍;含短 prompt(prompt ≤ N+1)场景。
- **采样模式有损**:与 MTP typical 接受不同,lookup 走 greedy 精确比较路径,采样参数(top_k/repeat_penalty 等)不支持时回退普通解码(门控与 MTP 一致)。
- **自适应停发**:per-request 滑窗统计接受率,低于阈值即停发(return false → 普通 ForwardGPU,CUDA graph 照常命中),每 PROBE_INTERVAL 个 token 探测恢复;停发与恢复窗口双清零,不震荡。
- **并发**:单后端单通道,多请求串行轮流(调度器单 lane),不做跨请求并行验证。
- `--lookup 8` 与 TP 快路径:草稿数 + 1 超过 `QWEN35_MTP_FAST_SEQ_MAX` 时验证走通用回退路径(与 MTP 同边界)。

## 统计日志

```
[Qwen3.5 Lookup] scheduler enabled: lookup_draft_max=4.
[Qwen3.5 Lookup] enabled: drafts_per_step=4, ngram=[2,4], max_entries=262144, ...
[Qwen3.5 Lookup] pos_accept_rate=[91.20%, 78.40%, ...]. hit_rate=45.10%(1234/2736).
```

`pos_accept_rate` 为各草稿位置接受率(与 MTP/DFlash 同口径);`hit_rate` 为查表命中率。每 64 次验证输出一行。

## 设计文档

OpenSpec change:`openspec/changes/add-prompt-lookup-draft/`(proposal / design / specs / tasks)。引擎组件(`include/prompt_lookup.h`)模型无关,可被其它 harness 复用;qwen3_5 为首个接入。

完整验证报告(两模型 base/lookup/MTP 对拍、速度表、翻转归因):[prompt-lookup-validation.md](./prompt-lookup-validation.md)。MTP 侧专题报告(历史脉络 + 本轮对照):[mtp-speculative-validation.md](./mtp-speculative-validation.md)。
