// prompt-lookup 引擎单元测试(spec: prompt-lookup-drafting 全部场景)。
// 独立测试目标:不链接 fastllm 库,证明引擎无模型依赖。
#include "prompt_lookup.h"

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using fastllm::PromptLookupIndex;

static void ExpectEq(const std::vector<int> &got, const std::vector<int> &want, const char *what) {
    if (got != want) {
        throw std::runtime_error(std::string(what) + ": result mismatch");
    }
}

static void ExpectTrue(bool ok, const char *what) {
    if (!ok) {
        throw std::runtime_error(std::string(what) + ": assertion failed");
    }
}

int main() {
    // 场景:prompt 含重复片段(末现覆盖)。窗口 [1,4]:A B C D A B C E 查 A B C → E(最近出现)。
    {
        PromptLookupIndex idx({1, 4, 262144});
        idx.Build({1, 2, 3, 4, 1, 2, 3, 5}); // A=1 B=2 C=3 D=4 E=5
        ExpectEq(idx.Lookup({1, 2, 3}, 4), {5}, "recency overwrite");
        // 确定性:重复查询一致。
        ExpectEq(idx.Lookup({1, 2, 3}, 4), idx.Lookup({1, 2, 3}, 4), "repeat query consistency");
    }

    // 场景:序列过短(无法形成任何 key)。
    {
        PromptLookupIndex idx;
        idx.Build({7});
        ExpectTrue(idx.Size() == 0, "too-short sequence yields empty index");
        ExpectTrue(idx.Lookup({7}, 4).empty(), "too-short sequence yields empty drafts");
    }

    // 场景:容量封顶保留近因(n_min=n_max=2,cap=10,100 个互不重复 token)。
    {
        PromptLookupIndex::Config cfg{2, 2, 10};
        PromptLookupIndex idx(cfg);
        std::vector<int> seq(100);
        std::iota(seq.begin(), seq.end(), 0);
        idx.Build(seq);
        ExpectTrue(idx.Size() == 10, "capacity cap enforced");
        ExpectTrue(idx.Lookup({0, 1}, 4).empty(), "ancient head entry evicted");
        ExpectEq(idx.Lookup({97, 98}, 4), {99}, "recent entry survives cap");
        // 追加继续驱逐最旧,容量守恒。
        idx.Append({100, 101, 102});
        ExpectTrue(idx.Size() == 10, "capacity conserved under append");
        ExpectEq(idx.Lookup({101, 102}, 4), {}, "appended tail beyond source end has no continuation");
        ExpectEq(idx.Lookup({100, 101}, 4), {102}, "appended entry present");
    }

    // 场景:增量追加等价于全量重建(含重复 token 的序列,多种切分)。
    {
        std::vector<int> seq = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9,
                                3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7, 9, 5, 5, 1};
        const size_t splits[] = {1, 5, 13, seq.size() - 1};
        for (size_t split : splits) {
            PromptLookupIndex full, incr;
            full.Build(seq);
            incr.Build(std::vector<int>(seq.begin(), seq.begin() + split));
            incr.Append(std::vector<int>(seq.begin() + split, seq.end()));
            ExpectTrue(full.Size() == incr.Size(), "incremental == full rebuild (size)");
            for (int q = 2; q < 12; q += 3) {
                std::vector<int> recent(seq.end() - q, seq.end());
                for (int n : {1, 4, 8}) {
                    ExpectEq(incr.Lookup(recent, n), full.Lookup(recent, n), "incremental == full rebuild");
                }
            }
        }
    }

    // 场景:长匹配优先于短匹配(3-gram 命中给 C..,错误实现走 1-gram 会给 Q..)。
    {
        PromptLookupIndex idx({1, 4, 262144});
        idx.Build({16, 1, 2, 3, 4, 1, 2, 17}); // P A B C D A B Q
        ExpectEq(idx.Lookup({16, 1, 2}, 4), {3, 4, 1, 2}, "longest match preferred");
    }

    // 场景:链长截断(连续续写超过上限 N → 恰取前 N)。
    {
        PromptLookupIndex idx({2, 2, 262144});
        idx.Build({1, 2, 3, 4, 5, 6, 7});
        ExpectEq(idx.Lookup({1, 2}, 3), {3, 4, 5}, "chain truncated to N");
        ExpectEq(idx.Lookup({1, 2}, 8), {3, 4, 5, 6, 7}, "chain natural length");
    }

    // 场景:延伸中途断链返回部分链(首段续写耗尽后无再命中 → 部分链,非空非全长)。
    {
        PromptLookupIndex idx({2, 4, 262144});
        idx.Build({1, 2, 3, 99});
        ExpectEq(idx.Lookup({1, 2}, 8), {3, 99}, "partial chain on mid-extension miss");
    }

    // 场景:延伸命中(首段耗尽后,后缀在别处再命中,续拼,且在 maxDrafts 处终止)。
    {
        PromptLookupIndex idx({2, 4, 262144});
        idx.Build({1, 2, 3, 4, 5, 1, 2, 3});
        std::vector<int> got = idx.Lookup({1, 2}, 8);
        ExpectEq(got, {3, 4, 5, 1, 2, 3, 4, 5}, "extension re-probe continues chain, capped at N");
    }

    // 场景:无任何命中 → 空。
    {
        PromptLookupIndex idx;
        idx.Build({1, 2, 3, 4});
        ExpectTrue(idx.Lookup({7, 7}, 4).empty(), "no match yields empty drafts");
        ExpectTrue(idx.Lookup({}, 4).empty(), "empty recent yields empty drafts");
    }

    // 场景:生成侧自我重复同样可加速(prompt 无重复,追加段含重复)。
    {
        PromptLookupIndex idx;
        idx.Build({10, 11, 12});
        idx.Append({5, 5, 6});
        ExpectEq(idx.Lookup({5, 5}, 2), {6}, "generated-side repetition accelerates");
    }

    // 配置护栏:ngramMin < 1、ngramMin > ngramMax 抛异常。
    {
        bool threw = false;
        try {
            PromptLookupIndex bad({0, 4, 100});
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        ExpectTrue(threw, "ngramMin < 1 rejected");
        threw = false;
        try {
            PromptLookupIndex bad({5, 3, 100});
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        ExpectTrue(threw, "ngramMin > ngramMax rejected");
    }

    std::cout << "prompt-lookup engine: PASS\n";
    return 0;
}
