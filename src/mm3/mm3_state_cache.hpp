#pragma once

#include "mm3_geo.hpp"
#include "mm3_kernels.cuh"
#include "mm3_model.hpp"
#include "strata/core/state_io.hpp"

#include <algorithm>
#include <filesystem>
#include <sstream>

namespace mm3 {
namespace statecache = strata::core::statecache;

inline std::string state_identity(const Pack& pack, const Geo& geometry, const Options& options,
                                  const std::string& layout) {
    std::ostringstream id;
    id << "mm3-state-1 " << layout << " " << std::filesystem::absolute(pack.dir).string()
       << " layers=" << geometry.n_layers << " heads=" << geometry.n_kv << " index=" << geometry.ix_dim
       << " k=" << options.kv_k << " v=" << options.kv_v << " rot=" << options.kv_rot
       << " topk=" << geometry.topk_override << " draft=" << (options.dspark && options.seqs == 1);
    for (const auto& item : pack.kv) id << " " << item.first << "=" << item.second;
    std::error_code error;
    for (const auto& file : pack.files) {
        const auto path = std::filesystem::path(pack.model_dir) / file;
        id << " " << file << ":" << std::filesystem::file_size(path, error) << ":"
           << std::filesystem::last_write_time(path, error).time_since_epoch().count();
    }
    return id.str();
}

inline void kv_state_regions(const std::string& name, int n, int heads, int index_width,
                             uint16_t* key, uint16_t* value, uint16_t* index,
                             const k::KvQ& quantized, const k::KvHost& host,
                             std::vector<statecache::MemoryRegion>& out) {
    const bool streamed = host.ring > 0;
    const auto& q = streamed ? host.q : quantized;
    uint16_t* bf16[] = {streamed ? host.kc : key, streamed ? host.vc : value};
    uint8_t* codes[] = {q.kc, q.vc};
    uint8_t* scales[] = {q.ks, q.vs};
    const int format[] = {q.kf, q.vf};
    const size_t rows = size_t(n) * heads;
    for (int role = 0; role < 2; ++role) {
        const std::string base = name + (role ? "/value" : "/key");
        if (format[role] == k::kKvBf16) out.push_back({base + "/bf16", {{bf16[role], rows * k::kv_code_bytes(format[role])}}});
        else {
            out.push_back({base + "/codes", {{codes[role], rows * k::kv_code_bytes(format[role])}}});
            out.push_back({base + "/scales", {{scales[role], rows * k::kv_scale_bytes(format[role])}}});
        }
    }
    out.push_back({name + "/index", {{index, size_t(n) * index_width * sizeof(uint16_t)}}});
}

struct StateDevice {
    int device = 0;
    std::vector<statecache::MemoryRegion> regions;
    std::vector<std::pair<int*, size_t>> tags;
};
inline bool sources_state_buffers(const std::string& identity, const std::vector<StateDevice>& devices,
                                  std::vector<statecache::RegionSource>& out, std::string& error) {
    out.clear(); out.push_back(statecache::bytes_source("layout", statecache::Bytes(identity.begin(), identity.end())));
    auto transfer = statecache::memory_transfer();
    for (const auto& device : devices) {
        cudaSetDevice(device.device);
        if (cudaDeviceSynchronize() != cudaSuccess ||
            !statecache::memory_sources(device.regions, device.device, transfer, out, error)) {
            if (error.empty()) error = "MiniMax capture producer failed";
            return false;
        }
    }
    return true;
}
inline bool load_state_buffers(const std::string& identity, const std::vector<StateDevice>& devices,
                               statecache::StateCache& cache, const statecache::StateCache::Lease& lease, std::string& error) {
    statecache::Bytes layout;
    if (!cache.read_region(lease, "layout", layout, error, identity.size()) ||
        std::string(layout.begin(), layout.end()) != identity) {
        if (error.empty()) error = "MiniMax state layout mismatch";
        return false;
    }
    auto transfer = statecache::memory_transfer();
    std::vector<statecache::RegionTarget> targets{statecache::check_region("layout", std::move(layout))};
    for (const auto& device : devices) {
        cudaSetDevice(device.device);
        for (const auto& tag : device.tags) if (cudaMemset(tag.first, 0xff, tag.second) != cudaSuccess) {
            error = "MiniMax residency invalidation failed"; return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess ||
            !statecache::memory_targets(device.regions, device.device, transfer, targets, error)) {
            if (error.empty()) error = "MiniMax restore producer failed";
            return false;
        }
    }
    return cache.restore(lease, targets, error);
}
inline bool capture_state_buffers(const std::string& identity, const statecache::Prefix& prefix,
                                  const std::vector<StateDevice>& devices, statecache::Capture& out,
                                  size_t budget, std::string& error) {
    statecache::Capture capture; capture.prefix = prefix;
    if (identity.size() > budget) { error = "MiniMax state exceeds host budget"; return false; }
    budget -= identity.size();
    capture.regions.push_back({"layout", statecache::Bytes(identity.begin(), identity.end())});
    for (const auto& device : devices) {
        cudaSetDevice(device.device);
        std::vector<statecache::Region> regions;
        if (!statecache::capture_memory(device.regions, regions, error, budget)) return false;
        for (auto& region : regions) { budget -= region.bytes.size(); capture.regions.push_back(std::move(region)); }
    }
    out = std::move(capture); return true;
}
inline bool restore_state_buffers(const std::string& identity, const std::vector<StateDevice>& devices,
                                  const std::vector<statecache::Region>& payload, std::string& error) {
    const auto layout = std::find_if(payload.begin(), payload.end(), [](const auto& r) { return r.name == "layout"; });
    if (layout == payload.end() || std::string(layout->bytes.begin(), layout->bytes.end()) != identity) {
        error = "MiniMax state layout mismatch"; return false;
    }
    bool ok = true;
    for (const auto& device : devices) {
        cudaSetDevice(device.device);
        // Ring rows may name another conversation. The host copy is authoritative;
        // invalidate residency even on a failed load before any reader can use it.
        for (const auto& tag : device.tags) if (cudaMemset(tag.first, 0xff, tag.second) != cudaSuccess) ok = false;
        if (!ok) continue;
        std::vector<statecache::Region> regions;
        for (const auto& view : device.regions) {
            const auto found = std::find_if(payload.begin(), payload.end(), [&](const auto& r) { return r.name == view.name; });
            if (found == payload.end()) { error = "MiniMax state member missing"; ok = false; break; }
            regions.push_back(*found);
        }
        if (ok) ok = statecache::restore_memory(device.regions, regions, error);
    }
    return ok;
}
}  // namespace mm3
