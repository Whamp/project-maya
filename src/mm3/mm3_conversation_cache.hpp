#pragma once

#include "mm3_model.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <memory>

namespace mm3 {
class ConversationCache {
    Model& model_;
    strata::core::statecache::Config config_;
    std::unique_ptr<strata::core::statecache::StateCache> cache_;
    std::string identity_;
public:
    explicit ConversationCache(Model& model) : model_(model),
        config_(strata::core::statecache::Config::environment(model.state_directory())), identity_(model.state_identity()) {
        const char* enabled = std::getenv("MAYA_STATE_CACHE");
        if (!enabled || !std::atoi(enabled)) return;
        try { cache_ = std::make_unique<strata::core::statecache::StateCache>(config_); }
        catch (const std::exception& e) { std::fprintf(stderr, "mm3 state cache: unavailable (%s)\n", e.what()); }
    }
    void observe(uint64_t from, uint64_t to, double seconds) { if (cache_) cache_->observe_prefill(from, to, seconds); }
    void report() const { if (cache_) std::printf("STATE %s\n", cache_->report().c_str()); }
    template <class Images>
    strata::core::statecache::Prefix prefix(const std::vector<int>& tokens, const Images& images) const {
        strata::core::statecache::Prefix result;
        result.identity = identity_; result.tokens.assign(tokens.begin(), tokens.end());
        for (const auto& image : images) if (image.start >= 0 && size_t(image.start) < tokens.size())
            result.extra.push_back({uint64_t(image.start), 1, std::to_string(image.hash)});
        return result;
    }
    template <class Images>
    void publish(const std::vector<int>& tokens, const Images& images, bool prompt = false) {
        if (!cache_ || model_.ablating() || tokens.empty()) return;
        try {
            for (int suffix = 0; suffix <= (prompt && tokens.size() > 1 ? 1 : 0); ++suffix) {
                auto state = prefix(tokens, images);
                state.tokens.resize(state.tokens.size() - size_t(suffix));
                state.extra.erase(std::remove_if(state.extra.begin(), state.extra.end(), [&](const auto& x) {
                    return x.position >= state.tokens.size();
                }), state.extra.end());
                strata::core::statecache::Capture capture;
                std::string error;
                capture.prefix = std::move(state);
                std::vector<strata::core::statecache::RegionSource> sources;
                if (model_.state_sources(capture.prefix, sources, error)) {
                    capture.frontier = !prompt;
                    cache_->publish(std::move(capture), sources, error);
                }
                if (!error.empty()) std::fprintf(stderr, "mm3 state cache: capture skipped (%s)\n", error.c_str());
            }
        } catch (const std::bad_alloc&) { std::fprintf(stderr, "mm3 state cache: host budget exhausted\n"); }
        report();
    }
    template <class Images>
    size_t restore(const std::vector<int>& prompt, const Images& images, size_t minimum, bool& changed) {
        changed = false;
        if (!cache_ || model_.ablating()) return minimum;
        auto lease = cache_->find_resume(prefix(prompt, images));
        if (!lease || lease.tokens() <= minimum) { cache_->release(lease); return minimum; }
        changed = true;
        std::string error;
        const bool ok = model_.state_load(*cache_, lease, error);
        const size_t endpoint = ok ? size_t(lease.tokens()) : 0;
        if (ok) cache_->restored(lease);
        cache_->release(lease);
        if (!ok) std::fprintf(stderr, "mm3 state cache: restore missed (%s); recomputing\n", error.c_str());
        return endpoint;
    }
};
}  // namespace mm3
