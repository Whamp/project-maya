#include "strata/core/session_state_cache.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cuda_runtime.h>
#include <algorithm>

namespace strata::core {
using statecache::MemoryRegion;
void session_running_regions(const ModelGeometry& g, SessionState& ss, const std::string& name,
                             std::vector<MemoryRegion>& out) {
    const size_t state = size_t(g.ssm_state_size) * g.ssm_v_heads * g.ssm_state_size;
    const size_t conv = size_t(g.ssm_conv_channels) * (g.ssm_d_conv - 1);
    out.push_back({name + "/gdn", {{ss.gdn_state, size_t(g.n_gdn_layers()) * (state + conv) * sizeof(float)}}});
    out.push_back({name + "/ple", {{ss.ple_hist, ss.ple_hist ? size_t(strata::kernels::NG_HIST) * strata::kernels::NG_HC_DIM * sizeof(float) : 0}}});
    MemoryRegion tails{name + "/tails", {}};
    const size_t bytes = size_t(strata::kernels::qsa_real_shapes().idx_block - 1) * g.idx_key_dim * sizeof(float);
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) tails.segments.push_back({ss.qsa_states[i].idx_tail, bytes});
    out.push_back(std::move(tails));
}

bool qsa_state_regions(const ModelGeometry& g, QsaState& st, int64_t n, const std::string& name,
                       bool indexer, std::vector<MemoryRegion>& out, std::string& error, int64_t first_cell) {
    auto qs = strata::kernels::qsa_real_shapes();
    qs.n_head_kv = g.n_head_kv; qs.head_dim = g.head_dim; qs.idx_dim = g.idx_key_dim;
    if (n <= 0 || n > st.max_cells) { error = "QSA endpoint exceeds cache capacity"; return false; }
    const int64_t pages = (n + qs.page_size - 1) / qs.page_size;
    std::vector<int32_t> table(size_t(pages), 0);
    if (!st.kv_mode && cudaMemcpy(table.data(), st.page_table, size_t(pages) * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        error = "cannot read QSA page mapping"; return false;
    }
    const auto pool = [&](const char* suffix, void* device, void* host, size_t row_bytes) {
        MemoryRegion r{name + suffix, {}};
        auto* base = static_cast<uint8_t*>(st.kv_mode ? host : device);
        for (int64_t pg = std::max<int64_t>(0, first_cell) / qs.page_size; pg < pages; ++pg) {
            const int64_t physical = st.kv_mode ? pg : table[size_t(pg)];
            if (physical < 0 || (!st.kv_mode && physical >= st.n_slots) || !base) return false;
            const int64_t begin = std::max<int64_t>(0, first_cell - pg * qs.page_size);
            const size_t cells = size_t(std::min(qs.page_size, n - pg * qs.page_size) - begin);
            for (int64_t head = 0; head < qs.n_head_kv; ++head) {
                const size_t offset = size_t((physical * qs.n_head_kv + head) * qs.page_size + begin) * row_bytes;
                r.segments.push_back({base + offset, cells * row_bytes});
            }
        }
        out.push_back(std::move(r)); return true;
    };
    const size_t q4 = strata::kernels::kv_q4_bytes_per_head(int(qs.head_dim));
    bool ok;
    if (st.kv_q4) ok = pool("/k4", st.k_q4, st.host.k_q4, q4) && pool("/v4", st.v_q4, st.host.v_q4, q4);
    else if (st.kv_hybrid) ok = pool("/k8", st.k_q, st.host.k_q, size_t(qs.head_dim)) &&
        pool("/ks", st.k_scale, st.host.k_scale, size_t(qs.head_dim / 64) * 2) && pool("/v4", st.v_q4, st.host.v_q4, q4);
    else if (st.kv_int8) ok = pool("/k8", st.k_q, st.host.k_q, size_t(qs.head_dim)) && pool("/v8", st.v_q, st.host.v_q, size_t(qs.head_dim)) &&
        pool("/ks", st.k_scale, st.host.k_scale, size_t(qs.head_dim / 64) * 2) && pool("/vs", st.v_scale, st.host.v_scale, size_t(qs.head_dim / 64) * 2);
    else ok = !st.kv_mode && pool("/k16", st.k_pool, nullptr, size_t(qs.head_dim) * 2) && pool("/v16", st.v_pool, nullptr, size_t(qs.head_dim) * 2);
    if (!ok) { error = "incomplete QSA physical backing"; return false; }
    if (indexer) {
        // The row for the unfinished block moves. Save only completed rows and
        // the sequence's constant dead key, then reconstruct that row on restore.
        out.push_back({name + "/pooled", {{st.idx_pooled, size_t(n / qs.idx_block) * qs.idx_dim * 4}}});
        out.push_back({name + "/dead", {{st.idx_dead, size_t(qs.idx_dim) * 4}}});
    }
    return true;
}
bool qsa_state_loaded(const ModelGeometry& g, QsaState& st, int64_t n, std::string& error) {
    auto qs = strata::kernels::qsa_real_shapes();
    qs.n_head_kv = g.n_head_kv; qs.head_dim = g.head_dim; qs.idx_dim = g.idx_key_dim;
    if (st.kv_mode == 1) strata::kernels::kv_stream_reset(st.map, nullptr);
    if (st.idx_pooled_rows > n / qs.idx_block &&
        cudaMemcpy(st.idx_pooled + (n / qs.idx_block) * qs.idx_dim, st.idx_dead, size_t(qs.idx_dim) * 4,
                   cudaMemcpyDeviceToDevice) != cudaSuccess) {
        error = "cannot reconstruct QSA unfinished-pool row"; return false;
    }
    if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(nullptr) != cudaSuccess) {
        error = "QSA residency reset failed"; return false;
    }
    return true;
}
}  // namespace strata::core
