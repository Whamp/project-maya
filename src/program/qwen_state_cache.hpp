#pragma once

#include "strata/core/mtp.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/session_state_cache.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

namespace strata::program {

// Policy belongs to StateCache; this adapter describes Qwen's physical state.
class QwenStateCache {
public:
    struct Stage { int device; core::SessionState* session; };
    struct Running { const core::statecache::Bytes *gdn, *ple, *tails; };
    using Bytes = core::statecache::Bytes;
    using Prefix = core::statecache::Prefix;
private:
    const core::ModelGeometry& g_;
    std::vector<Stage> stages_;
    core::MtpDrafter& mtp_;
    core::statecache::Config config_;
    core::statecache::StateCache cache_;
    Bytes residual_;
    int64_t residual_at_ = -1;

    bool memory(size_t stage, int64_t n, std::vector<core::statecache::MemoryRegion>& out, std::string& error) {
        const std::string name = "stage-" + std::to_string(stage);
        auto& session = *stages_[stage].session;
        core::session_running_regions(g_, session, name, out);
        for (int64_t layer = 0; layer < g_.n_qsa_layers(); ++layer)
            if (!core::qsa_state_regions(g_, session.qsa_states[layer], n,
                    name + "/attention-" + std::to_string(layer), true, out, error)) return false;
        return true;
    }
    void reset() {
        for (auto& stage : stages_) {
            const core::OnDevice on(stage.device);
            core::session_zero(*stage.session, g_, nullptr, nullptr);
            cudaDeviceSynchronize();
        }
        const core::OnDevice on(mtp_.device());
        core::qsa_state_zero(mtp_.kv_state(), g_, nullptr);
        cudaDeviceSynchronize();
    }
public:
    QwenStateCache(const core::ModelGeometry& g, std::vector<Stage> stages,
                   core::MtpDrafter& mtp, core::statecache::Config config)
        : g_(g), stages_(std::move(stages)), mtp_(mtp), config_(std::move(config)), cache_(config_) {}

    void observe(uint64_t from, uint64_t to, double seconds) { cache_.observe_prefill(from, to, seconds); }
    void report() const { std::printf("STATE %s\n", cache_.report().c_str()); }
    void remember(int64_t n, const float* residual) {
        residual_at_ = -1;
        const core::OnDevice on(mtp_.device());
        residual_.resize(size_t(g_.hc * g_.n_embd) * sizeof(float));
        if (residual && cudaMemcpy(residual_.data(), residual, residual_.size(), cudaMemcpyDeviceToHost) == cudaSuccess)
            residual_at_ = n;
    }

    bool publish(Prefix prefix, const std::vector<Running>& running, std::string& error, bool frontier = false) {
        namespace sc = core::statecache;
        const int64_t n = int64_t(prefix.tokens.size());
        if (n != residual_at_ || running.size() != stages_.size()) return true;
        sc::Capture capture; capture.prefix = std::move(prefix); capture.frontier = frontier;
        capture.min_prompt_tokens = uint64_t(mtp_.minimum_reusable_prompt());
        std::vector<sc::RegionSource> sources;
        auto transfer = sc::memory_transfer();
        for (size_t i = 0; i < stages_.size(); ++i) {
            const core::OnDevice on(stages_[i].device);
            std::vector<sc::MemoryRegion> regions;
            if (!memory(i, n, regions, error)) return false;
            // Each pipeline stage owns its running checkpoint at n. Positional
            // attention is append-only, so only its written prefix is described.
            const auto& r = running[i];
            regions[0].segments = {{const_cast<uint8_t*>(r.gdn->data()), r.gdn->size()}};
            regions[1].segments = {{const_cast<uint8_t*>(r.ple->data()), r.ple->size()}};
            regions[2].segments = {{const_cast<uint8_t*>(r.tails->data()), r.tails->size()}};
            const size_t begin = sources.size();
            if (!sc::memory_sources(regions, stages_[i].device, transfer, sources, error)) return false;
            auto& tails = sources[begin + 2];
            auto read = tails.read;
            const size_t row = size_t(g_.idx_key_dim) * sizeof(float);
            const size_t tail = size_t(strata::kernels::qsa_real_shapes().idx_block - 1) * row;
            const size_t valid = size_t(n % strata::kernels::qsa_real_shapes().idx_block) * row;
            tails.read = [read, tail, valid](uint64_t at, uint8_t* dst, size_t count, std::string& e) {
                if (!read(at, dst, count, e)) return false;
                for (size_t j = 0; j < count; ++j) if ((at+j)%tail >= valid) dst[j] = 0;
                return true;
            };
        }
        const core::OnDevice on(mtp_.device());
        if (!mtp_.idle(error)) return false;
        std::vector<sc::MemoryRegion> regions;
        // Cell n-1 consumes token n. Save [first,n-1), then rebuild that cell
        // from the endpoint residual and the next prompt's actual token.
        const int64_t first = std::min(n - 1, std::max<int64_t>(0, mtp_.first_needed()));
        if (n > 1 && !core::qsa_state_regions(g_, mtp_.kv_state_rw(), n - 1, "drafter", false, regions, error, first)) return false;
        if (!sc::memory_sources(regions, mtp_.device(), transfer, sources, error)) return false;
        sources.push_back(sc::bytes_source("drafter-residual", residual_));
        Bytes start(sizeof(first));
        std::memcpy(start.data(), &first, sizeof(first));
        sources.push_back(sc::bytes_source("drafter-first", std::move(start)));
        const bool ok = cache_.publish(std::move(capture), sources, error);
        report();
        return ok;
    }

    // A failed copy invalidates the live state as well: the caller recomputes.
    bool restore(const Prefix& prompt, int64_t& endpoint, bool& from_live, std::string& error) {
        namespace sc = core::statecache;
        auto lease = cache_.find_resume(prompt);
        if (!lease || int64_t(lease.tokens()) <= endpoint) { cache_.release(lease); return true; }
        const int64_t n = int64_t(lease.tokens());
        Bytes start, residual;
        int64_t first = -1;
        bool ok = cache_.read_region(lease, "drafter-first", start, error, sizeof(first)) && start.size() == sizeof(first) &&
                  cache_.read_region(lease, "drafter-residual", residual, error, size_t(g_.hc * g_.n_embd) * sizeof(float)) &&
                  residual.size() == size_t(g_.hc * g_.n_embd) * sizeof(float);
        if (ok) std::memcpy(&first, start.data(), sizeof(first));
        ok = ok && first >= 0 && first <= n - 1;
        {
            auto transfer = sc::memory_transfer();
            std::vector<sc::RegionTarget> targets{
                sc::check_region("drafter-first", start), sc::check_region("drafter-residual", residual)};
            for (size_t i = 0; ok && i < stages_.size(); ++i) {
                std::vector<sc::MemoryRegion> regions;
                ok = memory(i, n, regions, error) && sc::memory_targets(regions, stages_[i].device, transfer, targets, error);
            }
            std::vector<sc::MemoryRegion> regions;
            if (ok && n > 1) ok = core::qsa_state_regions(g_, mtp_.kv_state_rw(), n - 1, "drafter", false, regions, error, first);
            if (ok) ok = sc::memory_targets(regions, mtp_.device(), transfer, targets, error) && cache_.restore(lease, targets, error);
        }
        for (size_t i = 0; ok && i < stages_.size(); ++i) {
            const core::OnDevice on(stages_[i].device);
            auto& session = *stages_[i].session;
            for (int64_t layer = 0; ok && layer < g_.n_qsa_layers(); ++layer)
                ok = core::qsa_state_loaded(g_, session.qsa_states[layer], n, error);
            session.ple_prev[0] = n >= 2 ? lease.prefix().tokens[size_t(n - 2)] : -1;
            session.ple_prev[1] = lease.prefix().tokens[size_t(n - 1)];
        }
        if (ok) {
            const core::OnDevice on(mtp_.device());
            float* device_residual = nullptr;
            ok = cudaMalloc(&device_residual, residual.size()) == cudaSuccess;
            if (ok) ok = cudaMemcpy(device_residual, residual.data(), residual.size(), cudaMemcpyHostToDevice) == cudaSuccess;
            if (ok) {
                mtp_.set_prompt_len(int64_t(prompt.tokens.size()));
                mtp_.kv_restore(n - 1);
                const int32_t token = prompt.tokens[size_t(n)];
                ok = mtp_.prefill(device_residual, &token, 1, n - 1, error);
            }
            if (device_residual) cudaFree(device_residual);
        }
        if (ok) cache_.restored(lease);
        cache_.release(lease);
        if (!ok) { reset(); endpoint = 0; from_live = false; return false; }
        endpoint = n; from_live = true; return true;
    }
};
}  // namespace strata::program
