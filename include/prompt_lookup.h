#ifndef FASTLLM_PROMPT_LOOKUP_H
#define FASTLLM_PROMPT_LOOKUP_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

namespace fastllm {

// 模型无关的 n-gram prompt-lookup 草稿引擎(spec: prompt-lookup-drafting)。
// 纯 token 空间:不依赖任何模型代码/权重/设备,可被任意推测 harness 复用。
//
// 语义要点(与 spec 一一对应):
// - key = 位置前 n 个 token(n ∈ [ngramMin, ngramMax]),value = 源序列位置,
//   续写 = 该位置起的连续切片(PLD 式整段拷贝,不做逐 token 马尔可夫游走)
// - 重复 key 末现覆盖(recency):增量 Append 与全量 Build 语义等价
// - 容量封顶按"最近更新存活"驱逐:丢弃头部古旧条目,保留近因条目
// - 查询最长匹配优先;续写不足上限时以已产出续写为后缀再探测延伸;
//   延伸中途全 n 无命中返回当前部分链;初始即无命中返回空
class PromptLookupIndex {
public:
    struct Config {
        int ngramMin = 2;
        int ngramMax = 4;
        size_t maxEntries = 262144;
    };

    PromptLookupIndex() : PromptLookupIndex(Config()) {}
    explicit PromptLookupIndex(const Config &config);

    // 全量重建。等价于对同一序列分段 Append 的最终状态。
    void Build(const std::vector<int> &tokens);

    // 增量追加(追加源限定为已发布/已提交 token,由调用方保证)。
    void Append(const std::vector<int> &tokens);

    // recentTokens:最近的上下文 token(通常为请求 allTokens 尾部);
    // maxDrafts:草稿上限(1..8)。
    std::vector<int> Lookup(const std::vector<int> &recentTokens, int maxDrafts) const;

    size_t Size() const { return entries_.size(); }
    size_t SourceLength() const { return source_.size(); }

private:
    void InsertAt(size_t pos); // 为 source_[pos) 插入全部 n 的 key(末现覆盖/封顶驱逐)

    Config config_;
    std::vector<int> source_; // token 历史,续写切片的来源

    struct Entry {
        size_t pos;    // 续写起点(插入时 < source_.size(),source 只增不减)
        uint64_t tick; // 最近更新序号,驱动封顶驱逐(保近因)
    };
    struct KeyHash {
        size_t operator()(const std::vector<int> &key) const;
    };
    std::unordered_map<std::vector<int>, Entry, KeyHash> entries_;
    std::map<uint64_t, std::vector<int>> lru_; // tick -> key,驱逐最旧
    uint64_t tick_ = 0;
};

} // namespace fastllm

#endif
