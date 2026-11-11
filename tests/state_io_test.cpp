#include "strata/core/session_state_cache.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <filesystem>
#include <chrono>

namespace sc = strata::core::statecache;
namespace core = strata::core;

void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try {
        std::string error;
        // Larger than both staging buffers: exercise reuse after completion, an
        // overlaid view with an offset, and validation before any destination write.
        const size_t n = 19 << 20;
        uint8_t* device = nullptr;
        require(cudaMalloc(&device, n + 31) == cudaSuccess, "allocate transfer fixture");
        sc::Bytes bytes(n);
        for (size_t i = 0; i < n; ++i) bytes[i] = uint8_t(i * 37 + i / 65536);
        require(cudaMemcpy(device + 17, bytes.data(), n, cudaMemcpyHostToDevice) == cudaSuccess, "seed fixture");
        std::vector<sc::MemoryRegion> memory{{"overlay", {{device + 17, n}}}};
        std::vector<sc::Region> saved, out;
        require(!sc::capture_memory(memory, out, error, n - 1) && out.empty(), "reject capture before over-budget allocation");
        require(sc::capture_memory(memory, saved, error, n) && saved[0].bytes == bytes, "double-buffered capture preserves view");
        cudaMemset(device, 99, n + 31);
        auto malformed = saved; malformed[0].bytes.pop_back();
        require(!sc::restore_memory(memory, malformed, error), "refuse incomplete payload");
        uint8_t before = 0; cudaMemcpy(&before, device + 17, 1, cudaMemcpyDeviceToHost);
        require(before == 99, "layout rejection did not modify live state");
        require(sc::restore_memory(memory, saved, error) && sc::capture_memory(memory, out, error), "restore overlaid view");
        require(out[0].bytes == bytes, "overlaid restore exact");
        const auto directory = std::filesystem::temp_directory_path() /
            ("maya-state-gpu-stream-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        {
            sc::Config config; config.directory = directory.string(); config.ram_bytes = 1 << 20; config.disk_bytes = 32 << 20;
            sc::StateCache cache(config);
            memory[0].segments = {{device+17, (5 << 20)+13}, {device+17+(5 << 20)+13, n-(5 << 20)-13}};
            {
                auto transfer = sc::memory_transfer();
                std::vector<sc::RegionSource> sources;
                require(sc::memory_sources(memory, 0, transfer, sources, error), error);
                sc::Capture metadata; metadata.prefix = {"gpu-view", {1,2}, {}};
                require(cache.publish(std::move(metadata), sources, error), error);
            }
            cache.flush();
            require(cache.stats().ram_bytes == 0 && cache.stats().disk_bytes > config.ram_bytes,
                    "GPU state exceeding host budget streams to SSD");
            require(cudaMemset(device, 99, n+31) == cudaSuccess, "overwrite streamed state source");
            auto lease = cache.find_resume({"gpu-view", {1,2,3}, {}});
            {
                auto transfer = sc::memory_transfer();
                std::vector<sc::RegionTarget> targets;
                require(sc::memory_targets(memory, 0, transfer, targets, error), error);
                require(lease && cache.restore(lease, targets, error), error);
            }
            cache.restored(lease); cache.release(lease); cache.flush();
            require(sc::capture_memory(memory, out, error) && out[0].bytes == bytes,
                    "bounded GPU restore preserves every overlaid byte across segment boundaries");
        }
        std::filesystem::remove_all(directory);
        cudaFree(device);
        // Real QSA layouts: permuted resident pages, all supported dtypes, and a
        // partial final page. Captures never include the unwritten tail capacity.
        for (int format = 0; format < 4; ++format) {
            core::qsa_set_kv_int8(format == 1);
            core::qsa_set_kv_q4(format == 2);
            core::qsa_set_kv_hybrid(format == 3);
            core::ModelGeometry geometry;
            const size_t size = core::qsa_state_bytes(geometry, 16);
            void* arena = nullptr;
            require(cudaMalloc(&arena, size) == cudaSuccess, "allocate QSA fixture");
            core::QsaState state;
            require(core::qsa_state_init(geometry, 16, arena, state) != 0, "initialize QSA fixture");
            const int32_t pages[] = {2, 0, 3, 1};
            require(cudaMemcpy(state.page_table, pages, sizeof(pages), cudaMemcpyHostToDevice) == cudaSuccess, "permute pages");
            memory.clear(); saved.clear(); out.clear();
            require(core::qsa_state_regions(geometry, state, 7, "qsa", true, memory, error), error);
            size_t serial = 0;
            for (auto& region : memory) for (auto& segment : region.segments) {
                sc::Bytes seed(segment.bytes);
                for (size_t i = 0; i < seed.size(); ++i) seed[i] = uint8_t(i * 13 + serial++);
                require(cudaMemcpy(segment.address, seed.data(), seed.size(), cudaMemcpyHostToDevice) == cudaSuccess, "seed physical view");
            }
            require(sc::capture_memory(memory, saved, error), error);
            for (auto& region : memory) for (auto& segment : region.segments)
                require(cudaMemset(segment.address, 0, segment.bytes) == cudaSuccess, "overwrite live cells");
            require(sc::restore_memory(memory, saved, error) && sc::capture_memory(memory, out, error), error);
            require(out.size() == saved.size(), "complete QSA member set");
            for (size_t i = 0; i < out.size(); ++i) require(out[i].bytes == saved[i].bytes, "QSA layout roundtrip");
            // These one-time QSA allocations use pinned step/position staging.
            cudaFreeHost(state.host_step); cudaFreeHost(state.host_pos);
            cudaFree(arena);
        }
        std::puts("state IO: bounded staging, overlaid views, invalid payloads and four QSA formats passed");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
