#include "glm_metal_impl.hpp"
#include "mm3_metal_impl.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <sys/stat.h>

namespace maya::metal {
namespace sc = strata::core::statecache;
namespace {
struct Slice { std::string name; ggml_tensor* tensor; size_t offset, bytes; };
void add(std::vector<Slice>& slices, const std::string& name, ggml_tensor* t, size_t offset, size_t bytes) {
    if (t && bytes) slices.push_back({name, t, offset, bytes});
}
void whole(std::vector<Slice>& slices, const std::string& name, ggml_tensor* t) {
    if (t) add(slices, name, t, 0, ggml_nbytes(t));
}
std::string identity(const std::string& kind, const std::string& path, const std::vector<int>& fds,
                     const std::vector<Slice>& slices) {
    std::ostringstream s;
    s << "ggml-state-v1:" << kind << ':' << std::filesystem::absolute(path).string();
    for (int fd : fds) {
        struct stat st{};
        if (fstat(fd, &st)) throw std::runtime_error("cannot fingerprint model shard");
        s << ':' << st.st_size << ':' << st.st_mtime;
#ifdef __APPLE__
        s << ':' << st.st_mtimespec.tv_nsec;
#else
        s << ':' << st.st_mtim.tv_nsec;
#endif
    }
    for (const auto& x : slices) {
        s << ':' << x.name << ':' << x.tensor->type;
        for (auto n : x.tensor->ne) s << ':' << n;
    }
    const auto text = s.str(); return sc::content_digest(text.data(), text.size());
}
bool capture(sc::Capture& out, const sc::Prefix& prefix, const std::vector<Slice>& slices,
             uint64_t budget, std::string& error) {
    out = {}; out.prefix = prefix;
    size_t total = 0;
    for (const auto& x : slices) {
        if (x.bytes > budget - total) { error = "Metal state exceeds host budget"; return false; }
        total += x.bytes;
    }
    for (const auto& x : slices) {
        sc::Region region{x.name, sc::Bytes(x.bytes)};
        ggml_backend_tensor_get(x.tensor, region.bytes.data(), x.offset, x.bytes);
        out.regions.push_back(std::move(region));
    }
    return true;
}
bool validate(const std::vector<sc::Region>& regions, const std::vector<Slice>& slices,
              size_t extra, std::string& error) {
    if (regions.size() != slices.size() + extra) { error = "incomplete Metal recovery unit"; return false; }
    for (size_t i = 0; i < slices.size(); ++i)
        if (regions[i].name != slices[i].name || regions[i].bytes.size() != slices[i].bytes) {
            error = "Metal state layout mismatch: " + slices[i].name; return false;
        }
    return true;
}
void restore(const std::vector<sc::Region>& regions, const std::vector<Slice>& slices) {
    for (size_t i = 0; i < slices.size(); ++i)
        ggml_backend_tensor_set(slices[i].tensor, regions[i].bytes.data(), slices[i].offset, slices[i].bytes);
}
void sources(const std::vector<Slice>& slices, std::vector<sc::RegionSource>& out) {
    for (const auto& slice : slices) out.push_back({slice.name, slice.bytes,
        [slice](uint64_t at, uint8_t* data, size_t n, std::string& error) {
            if (at > slice.bytes || n > slice.bytes-at) { error = "Metal capture view out of range"; return false; }
            ggml_backend_tensor_get(slice.tensor, data, slice.offset+size_t(at), n); return true;
        }});
}
void targets(const std::vector<Slice>& slices, std::vector<sc::RegionTarget>& out) {
    for (const auto& slice : slices) out.push_back({slice.name, slice.bytes,
        [slice](uint64_t at, const uint8_t* data, size_t n, std::string& error) {
            if (at > slice.bytes || n > slice.bytes-at) { error = "Metal restore view out of range"; return false; }
            ggml_backend_tensor_set(slice.tensor, data, slice.offset+size_t(at), n); return true;
        }});
}
void sideband(const std::string& name, sc::Bytes bytes, std::vector<sc::RegionSource>& out) {
    auto owned = std::make_shared<const sc::Bytes>(std::move(bytes));
    out.push_back({name, owned->size(), [owned](uint64_t at, uint8_t* data, size_t n, std::string& error) {
        if (at > owned->size() || n > owned->size()-at) { error = "Metal sideband out of range"; return false; }
        std::memcpy(data, owned->data()+at, n); return true;
    }});
}
void sideband(const std::string& name, const sc::Bytes& bytes, std::vector<sc::RegionTarget>& out) {
    out.push_back({name, bytes.size(), [&bytes](uint64_t at, const uint8_t* data, size_t n, std::string& error) {
        if (at > bytes.size() || n > bytes.size()-at || std::memcmp(data, bytes.data()+at, n)) {
            error = "Metal sideband changed during restore"; return false;
        }
        return true;
    }});
}
std::vector<Slice> glm_slices(const GlmMetal::Impl& m, size_t n, bool snap, bool drafter = true) {
    std::vector<Slice> out;
    for (size_t i = 0; i < m.ls.size(); ++i) {
        const auto& s = m.ls[i]; const auto pre = "layer/" + std::to_string(i) + '/';
        whole(out, pre + "conv", snap ? s.conv_snap : s.conv);
        whole(out, pre + "ssm", snap ? s.ssm_snap : s.ssm);
        // NextN's row p pairs hidden[p] with input[p+1]. The last row is
        // deliberately reconstructed by the next forward, never saved stale.
        const size_t written = i == size_t(m.g.n_layers) ? n - 1 : n;
        const size_t pooled = written / size_t(m.g.idx_kpool);
        const size_t open = pooled * size_t(m.g.idx_kpool);
        if (s.lat) add(out, pre + "latent", s.lat, 0, written * s.lat->nb[1]);
        if (s.ikg) add(out, pre + "open-pool", s.ikg, open * s.ikg->nb[1], (written - open) * s.ikg->nb[1]);
        if (s.pooled) add(out, pre + "pooled", s.pooled, 0, pooled * s.pooled->nb[1]);
    }
    if (m.df && drafter) for (size_t i = 0; i < m.df->ring_k.size(); ++i)
        for (size_t slot = 0; slot < m.df->kv_pos.size(); ++slot)
            if (m.df->kv_pos[slot] >= 0 && size_t(m.df->kv_pos[slot]) < n) {
                const auto pre = "dflash/" + std::to_string(i) + '/' + std::to_string(slot);
                add(out, pre + "/k", m.df->ring_k[i], slot * m.df->ring_k[i]->nb[2], m.df->ring_k[i]->nb[2]);
                add(out, pre + "/v", m.df->ring_v[i], slot * m.df->ring_v[i]->nb[2], m.df->ring_v[i]->nb[2]);
            }
    return out;
}
std::vector<Slice> mm3_slices(const Mm3Metal::Impl& m, size_t n) {
    std::vector<Slice> out;
    for (size_t i = 0; i < m.ls.size(); ++i) {
        const auto& s = m.ls[i]; const auto pre = "layer/" + std::to_string(i) + '/';
        add(out, pre + "k", s.kc, 0, n * s.kc->nb[2]);
        add(out, pre + "v", s.vc, 0, n * s.vc->nb[2]);
        add(out, pre + "index", s.ikc, 0, n * s.ikc->nb[1]);
    }
    return out;
}
}

std::string GlmMetal::state_identity() const {
    const auto& m = *p_;
    auto slices = glm_slices(m, 1, false, false);
    auto files = m.fds;
    if (m.df) {
        files.push_back(m.df->fd);
        for (size_t i = 0; i < m.df->ring_k.size(); ++i) {
            whole(slices, "dflash/k/" + std::to_string(i), m.df->ring_k[i]);
            whole(slices, "dflash/v/" + std::to_string(i), m.df->ring_v[i]);
        }
    }
    return identity("glm:" + std::to_string(m.o.use_gpu) + ':' + std::to_string(m.o.dsa_mm) + ':' +
                    std::to_string(m.o.dsa_kernel) + ':' + std::to_string(m.o.n_ubatch) + ':' + m.o.dflash,
                    m.model_path, files, slices);
}
std::string GlmMetal::state_directory() const { return p_->model_path + ".state-cache"; }
bool GlmMetal::state_sources(const sc::Prefix& prefix, std::vector<sc::RegionSource>& out, std::string& error) const {
    const auto& m = *p_; const size_t n = prefix.tokens.size();
    if (!n || n > size_t(m.pos) || (n != size_t(m.pos) && n != size_t(m.snap_pos))) {
        error = "Metal recurrence endpoint unavailable"; return false;
    }
    const bool snap = n != size_t(m.pos);
    if (m.df && snap) { error = "DFlash snapshot history displaced"; return false; }
    ggml_backend_sched_synchronize(m.sched);
    out.clear(); sources(glm_slices(m, n, snap), out);
    const auto& hidden = snap ? m.hid_snap : m.hid_host;
    const size_t offset = snap ? 0 : size_t(m.hx_col) * m.g.n_embd;
    if (hidden.size() < offset+size_t(m.g.n_embd)) { error = "Metal endpoint hidden state unavailable"; return false; }
    sc::Bytes h(size_t(m.g.n_embd)*4);
    std::memcpy(h.data(), hidden.data()+offset, h.size()); sideband("endpoint-hidden", std::move(h), out);
    if (m.df) {
        auto positions = m.df->kv_pos;
        for (auto& p : positions) if (p >= int64_t(n)) p = -1;
        sc::Bytes tags(positions.size()*8);
        std::memcpy(tags.data(), positions.data(), tags.size()); sideband("dflash-tags", std::move(tags), out);
    }
    return true;
}
bool GlmMetal::state_load(sc::StateCache& cache, const sc::StateCache::Lease& lease, std::string& error) {
    auto& m = *p_; const size_t n = size_t(lease.tokens());
    if (!n || n > size_t(m.n_ctx)) { error = "Metal endpoint out of range"; return false; }
    sc::Bytes hidden, tags;
    if (!cache.read_region(lease, "endpoint-hidden", hidden, error, size_t(m.g.n_embd)*4) ||
        hidden.size() != size_t(m.g.n_embd)*4) return false;
    if (m.df) {
        if (!cache.read_region(lease, "dflash-tags", tags, error, m.df->kv_pos.size()*8) ||
            tags.size() != m.df->kv_pos.size()*8) return false;
        std::vector<int64_t> positions(m.df->kv_pos.size());
        std::memcpy(positions.data(), tags.data(), tags.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            const auto p = positions[i];
            if (p < -1 || p >= int64_t(n) || (p >= 0 && size_t(p)%positions.size() != i)) {
                error = "invalid DFlash ring position"; return false;
            }
        }
        m.df->kv_pos = std::move(positions);
    }
    std::vector<sc::RegionTarget> destination;
    sideband("endpoint-hidden", hidden, destination);
    if (m.df) sideband("dflash-tags", tags, destination);
    targets(glm_slices(m, n, false), destination);
    ggml_backend_sched_synchronize(m.sched);
    if (!cache.restore(lease, destination, error)) { reset(); return false; }
    m.hid_host.resize(size_t(m.g.n_embd));
    std::memcpy(m.hid_host.data(), hidden.data(), hidden.size());
    ggml_backend_tensor_set(m.hid, m.hid_host.data(), 0, hidden.size());
    for (const auto& s : m.ls) if (s.lat)
        ggml_backend_tensor_memset(s.lat, 0, size_t(m.n_ctx)*s.lat->nb[1], s.lat->nb[1]);
    ggml_backend_sched_synchronize(m.sched);
    m.pos = int64_t(n); m.snap_pos = -1; m.win_T = 0; m.hx_pos = int64_t(n)-1; m.hx_col = 0; m.hid_rows = 1;
    m.mtp_hx.clear(); return true;
}
bool GlmMetal::state_capture(sc::Capture& out, const sc::Prefix& prefix, std::string& error) const {
    const auto& m = *p_; const size_t n = prefix.tokens.size();
    if (!n || n > size_t(m.pos) || (n != size_t(m.pos) && n != size_t(m.snap_pos))) {
        error = "Metal recurrence endpoint unavailable"; return false;
    }
    const bool snap = n != size_t(m.pos);
    // DFlash's rolling history can no longer represent an old snapshot after
    // it has wrapped. Publish that frontier when it is current instead.
    if (m.df && snap) { error = "DFlash snapshot history displaced"; return false; }
    ggml_backend_sched_synchronize(m.sched);
    if (!capture(out, prefix, glm_slices(m, n, snap), sc::Config::environment().ram_bytes, error)) return false;
    const auto& hidden = snap ? m.hid_snap : m.hid_host;
    const size_t offset = snap ? 0 : size_t(m.hx_col) * m.g.n_embd;
    if (hidden.size() < offset + size_t(m.g.n_embd)) { error = "Metal endpoint hidden state unavailable"; return false; }
    sc::Region h{"endpoint-hidden", sc::Bytes(size_t(m.g.n_embd) * sizeof(float))};
    std::memcpy(h.bytes.data(), hidden.data() + offset, h.bytes.size()); out.regions.push_back(std::move(h));
    if (m.df) {
        sc::Region tags{"dflash-tags", sc::Bytes(m.df->kv_pos.size() * sizeof(int64_t))};
        auto positions = m.df->kv_pos;
        for (auto& p : positions) if (p >= int64_t(n)) p = -1;
        std::memcpy(tags.bytes.data(), positions.data(), tags.bytes.size()); out.regions.push_back(std::move(tags));
    }
    return true;
}
bool GlmMetal::state_restore(const sc::Prefix& prefix, const std::vector<sc::Region>& regions, std::string& error) {
    auto& m = *p_; const size_t n = prefix.tokens.size();
    if (!n || n > size_t(m.n_ctx)) { error = "Metal endpoint out of range"; return false; }
    std::vector<int64_t> old_tags;
    if (m.df) {
        if (regions.empty() || regions.back().name != "dflash-tags" || regions.back().bytes.size() != m.df->kv_pos.size()*8) {
            error = "missing DFlash ring positions"; return false;
        }
        old_tags = m.df->kv_pos;
        std::memcpy(m.df->kv_pos.data(), regions.back().bytes.data(), regions.back().bytes.size());
        for (size_t i = 0; i < m.df->kv_pos.size(); ++i) {
            const auto p = m.df->kv_pos[i];
            if (p < -1 || p >= int64_t(n) || (p >= 0 && size_t(p) % m.df->kv_pos.size() != i)) {
                m.df->kv_pos = old_tags; error = "invalid DFlash ring position"; return false;
            }
        }
    }
    const auto slices = glm_slices(m, n, false);
    const size_t extra = m.df ? 2 : 1;
    if (!validate(regions, slices, extra, error) || regions[slices.size()].name != "endpoint-hidden" ||
        regions[slices.size()].bytes.size() != size_t(m.g.n_embd)*4) {
        if (m.df) m.df->kv_pos = old_tags;
        if (error.empty()) error = "missing Metal endpoint hidden state";
        return false;
    }
    ggml_backend_sched_synchronize(m.sched);
    restore(regions, slices);
    m.hid_host.resize(size_t(m.g.n_embd));
    std::memcpy(m.hid_host.data(), regions[slices.size()].bytes.data(), m.hid_host.size()*4);
    ggml_backend_tensor_set(m.hid, m.hid_host.data(), 0, m.hid_host.size()*4);
    m.pos = int64_t(n); m.snap_pos = -1; m.win_T = 0; m.hx_pos = int64_t(n)-1; m.hx_col = 0; m.hid_rows = 1;
    m.mtp_hx.clear();
    for (const auto& s : m.ls) if (s.lat)
        ggml_backend_tensor_memset(s.lat, 0, size_t(m.n_ctx) * s.lat->nb[1], s.lat->nb[1]);
    return true;
}

std::string Mm3Metal::state_identity() const {
    const auto& m = *p_;
    return identity("mm3:" + std::to_string(m.o.use_gpu), m.model_path, m.fds, mm3_slices(m, 1));
}
std::string Mm3Metal::state_directory() const { return p_->model_path + ".state-cache"; }
bool Mm3Metal::state_sources(const sc::Prefix& prefix, std::vector<sc::RegionSource>& out, std::string& error) const {
    const auto& m = *p_; const size_t n = prefix.tokens.size();
    if (!n || n > size_t(m.pos)) { error = "MiniMax Metal endpoint unwritten"; return false; }
    ggml_backend_sched_synchronize(m.sched); out.clear(); sources(mm3_slices(m, n), out); return true;
}
bool Mm3Metal::state_load(sc::StateCache& cache, const sc::StateCache::Lease& lease, std::string& error) {
    auto& m = *p_; const size_t n = size_t(lease.tokens());
    if (!n || n > size_t(m.n_ctx)) { error = "MiniMax Metal endpoint out of range"; return false; }
    std::vector<sc::RegionTarget> destination; targets(mm3_slices(m, n), destination);
    ggml_backend_sched_synchronize(m.sched);
    if (!cache.restore(lease, destination, error)) { reset(); return false; }
    ggml_backend_sched_synchronize(m.sched); m.pos = int64_t(n); m.snap_pos = -1; return true;
}
bool Mm3Metal::state_capture(sc::Capture& out, const sc::Prefix& prefix, std::string& error) const {
    const auto& m = *p_; const size_t n = prefix.tokens.size();
    if (!n || n > size_t(m.pos)) { error = "MiniMax Metal endpoint unwritten"; return false; }
    ggml_backend_sched_synchronize(m.sched);
    return capture(out, prefix, mm3_slices(m, n), sc::Config::environment().ram_bytes, error);
}
bool Mm3Metal::state_restore(const sc::Prefix& prefix, const std::vector<sc::Region>& regions, std::string& error) {
    auto& m = *p_; const size_t n = prefix.tokens.size();
    if (!n || n > size_t(m.n_ctx)) { error = "MiniMax Metal endpoint out of range"; return false; }
    const auto slices = mm3_slices(m, n);
    if (!validate(regions, slices, 0, error)) return false;
    ggml_backend_sched_synchronize(m.sched); restore(regions, slices);
    m.pos = int64_t(n); m.snap_pos = -1; return true;
}
}  // namespace maya::metal
