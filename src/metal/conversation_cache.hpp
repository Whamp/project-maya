#pragma once
#include "strata/core/state_cache.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace maya::metal {
template <class Model> class ConversationCache {
    Model& model_;
    std::string identity_;
    std::unique_ptr<strata::core::statecache::StateCache> cache_;
public:
    explicit ConversationCache(Model& model) : model_(model) {
        const char* on = std::getenv("MAYA_STATE_CACHE");
        if (!on || !std::atoi(on)) return;
        try {
            identity_ = model.state_identity();
            cache_ = std::make_unique<strata::core::statecache::StateCache>(
                strata::core::statecache::Config::environment(model.state_directory()));
        } catch (const std::exception& e) { std::fprintf(stderr, "state cache: unavailable (%s)\n", e.what()); }
    }
    strata::core::statecache::Prefix prefix(const std::vector<int32_t>& tokens,
        const std::vector<strata::core::statecache::ExtraInput>& images) const {
        strata::core::statecache::Prefix p{identity_, tokens, images};
        p.extra.erase(std::remove_if(p.extra.begin(), p.extra.end(), [&](const auto& x) {
            return x.position >= p.tokens.size();
        }), p.extra.end());
        return p;
    }
    void publish(const std::vector<int32_t>& tokens,
                 const std::vector<strata::core::statecache::ExtraInput>& images = {}, bool frontier = false) {
        if (!cache_ || tokens.empty()) return;
        try {
            strata::core::statecache::Capture state;
            std::string error;
            std::vector<strata::core::statecache::RegionSource> sources;
            state.prefix = prefix(tokens, images); state.frontier = frontier;
            if (model_.state_sources(state.prefix, sources, error)) {
                cache_->publish(std::move(state), sources, error);
            }
            if (!error.empty()) std::fprintf(stderr, "state cache: capture skipped (%s)\n", error.c_str());
        } catch (const std::bad_alloc&) { std::fprintf(stderr, "state cache: host budget exhausted\n"); }
    }
    size_t restore(const std::vector<int32_t>& prompt, size_t minimum,
                   const std::vector<strata::core::statecache::ExtraInput>& images = {}) {
        if (!cache_) return minimum;
        try {
            auto lease = cache_->find_resume(prefix(prompt, images));
            if (!lease || lease.tokens() <= minimum) { cache_->release(lease); return minimum; }
            std::string error;
            const bool ok = model_.state_load(*cache_, lease, error);
            const size_t n = ok ? size_t(lease.tokens()) : 0;
            if (ok) cache_->restored(lease);
            cache_->release(lease);
            if (!ok) { model_.reset(); std::fprintf(stderr, "state cache: restore missed (%s)\n", error.c_str()); }
            return n;
        } catch (const std::bad_alloc&) { model_.reset(); return 0; }
    }
    void observe(size_t from, size_t to, double seconds) { if (cache_) cache_->observe_prefill(from, to, seconds); }
    void report() const { if (cache_) std::printf("STATE %s\n", cache_->report().c_str()); }
};
}
