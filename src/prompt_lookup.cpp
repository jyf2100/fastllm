#include "prompt_lookup.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace fastllm {

PromptLookupIndex::PromptLookupIndex(const Config &config) : config_(config) {
    if (config_.ngramMin < 1) {
        throw std::invalid_argument("PromptLookupIndex: ngramMin must be >= 1");
    }
    if (config_.ngramMin > config_.ngramMax) {
        throw std::invalid_argument("PromptLookupIndex: ngramMin must be <= ngramMax");
    }
}

size_t PromptLookupIndex::KeyHash::operator()(const std::vector<int> &key) const {
    // FNV-1a over token ids;确定性与容器实现无关。
    size_t hash = 1469598103934665603ULL;
    for (int v : key) {
        hash ^= (size_t)(uint32_t)v;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void PromptLookupIndex::Build(const std::vector<int> &tokens) {
    entries_.clear();
    lru_.clear();
    tick_ = 0;
    source_.clear();
    Append(tokens);
}

void PromptLookupIndex::Append(const std::vector<int> &tokens) {
    size_t start = source_.size();
    source_.insert(source_.end(), tokens.begin(), tokens.end());
    if (config_.maxEntries == 0) {
        return;
    }
    if (entries_.empty()) {
        // 预留桶避免成长期 rehash 反复全量重哈希 vector key。
        size_t estimate = source_.size() * (size_t)(config_.ngramMax - config_.ngramMin + 1);
        entries_.reserve(std::min<size_t>(estimate, config_.maxEntries));
    }
    for (size_t pos = start; pos < source_.size(); pos++) {
        InsertAt(pos);
    }
}

void PromptLookupIndex::InsertAt(size_t pos) {
    int top = std::min<int>(config_.ngramMax, (int)pos);
    for (int n = config_.ngramMin; n <= top; n++) {
        std::vector<int> key(source_.begin() + pos - n, source_.begin() + pos);
        uint64_t tick = ++tick_;
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            // 末现覆盖:同 key 取最近出现位置,并刷新其驱逐序。
            lru_.erase(it->second.tick);
            it->second.pos = pos;
            it->second.tick = tick;
            lru_.emplace(tick, it->first);
        } else {
            if (entries_.size() >= config_.maxEntries && !lru_.empty()) {
                // 封顶驱逐最旧条目:保留近因,丢弃头部古旧区。
                auto oldest = lru_.begin();
                entries_.erase(oldest->second);
                lru_.erase(oldest);
            }
            auto inserted = entries_.emplace(std::move(key), Entry{pos, tick});
            lru_.emplace(tick, inserted.first->first);
        }
    }
}

std::vector<int> PromptLookupIndex::Lookup(const std::vector<int> &recentTokens, int maxDrafts) const {
    std::vector<int> drafts;
    if (maxDrafts <= 0 || entries_.empty() || source_.empty()) {
        return drafts;
    }
    // combined = 上下文 + 已产出草稿;延伸探测以其尾部 n 个 token 为 key。
    std::vector<int> combined(recentTokens);
    while ((int)drafts.size() < maxDrafts) {
        bool hit = false;
        size_t pos = 0;
        int top = std::min<int>(config_.ngramMax, (int)combined.size());
        for (int n = top; n >= config_.ngramMin; n--) {
            std::vector<int> key(combined.end() - n, combined.end());
            auto it = entries_.find(key);
            if (it != entries_.end()) {
                pos = it->second.pos;
                hit = true;
                break;
            }
        }
        if (!hit) {
            break;
        }
        size_t avail = source_.size() > pos ? source_.size() - pos : 0;
        size_t take = std::min<size_t>(avail, (size_t)maxDrafts - drafts.size());
        if (take == 0) {
            break;
        }
        drafts.insert(drafts.end(), source_.begin() + pos, source_.begin() + pos + take);
        if (take < avail) {
            break; // 已拿满 maxDrafts
        }
        // 续写耗尽(到源序列末尾):以含新草稿的后缀再探测延伸。
        combined.insert(combined.end(), source_.begin() + pos, source_.begin() + pos + take);
    }
    return drafts;
}

} // namespace fastllm
