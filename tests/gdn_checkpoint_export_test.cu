#include "strata/prefill/kernels.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_ple_postops.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

void need(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Buffer {
    void* data = nullptr;
    explicit Buffer(size_t bytes) { need(cudaMalloc(&data, bytes) == cudaSuccess, "allocate exporter fixture"); }
    ~Buffer() { cudaFree(data); }
    float* f() const { return static_cast<float*>(data); }
    uint16_t* h() const { return static_cast<uint16_t*>(data); }
    void seed(size_t n, int salt, float scale = .03f, float bias = 0) {
        std::vector<float> values(n);
        for (size_t i = 0; i < n; ++i) values[i] = bias + scale * std::sin(float((i * 13 + salt) % 1999));
        need(cudaMemcpy(data, values.data(), n * 4, cudaMemcpyHostToDevice) == cudaSuccess, "seed exporter fixture");
    }
    std::vector<uint8_t> read(size_t bytes) const {
        std::vector<uint8_t> result(bytes);
        need(cudaMemcpy(result.data(), data, bytes, cudaMemcpyDeviceToHost) == cudaSuccess, "read exporter fixture");
        return result;
    }
};
int main(int argc, char**) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    // Serial and column-split dispatch are selected once per process.
#ifdef _WIN32
    _putenv_s("MAYA_GDN_REC_HEADS", argc > 1 ? "1" : "");
    _putenv_s("MAYA_GDN_CONV_SERIAL", argc > 1 ? "1" : "");
#else
    if (argc > 1) { setenv("MAYA_GDN_REC_HEADS", "1", 1); setenv("MAYA_GDN_CONV_SERIAL", "1", 1); }
    else { unsetenv("MAYA_GDN_REC_HEADS"); unsetenv("MAYA_GDN_CONV_SERIAL"); }
#endif
    try {
        constexpr int T = 73, C = 10240, HV = 48, S = 128, V = HV * S;
        const size_t state_bytes = size_t(S) * HV * S * 4;
        cudaStream_t stream = nullptr;
        need(cudaStreamCreate(&stream) == cudaSuccess, "create exporter stream");
        Buffer state(state_bytes), saved(state_bytes), conv(C * 3 * 4), conv_saved(C * 3 * 4);
        Buffer raw(T * C * 4), weights(C * 4 * 4), h(T * C * 4);
        Buffer gates(T * HV * 4), beta(T * HV * 4), z(T * V * 4), gamma(S * 4), y(T * V * 4), y16(T * V * 2);
        raw.seed(T * C, 7); weights.seed(C * 4, 3); gates.seed(T * HV, 5, .01f, -.08f);
        beta.seed(T * HV, 2, .02f, .45f); z.seed(T * V, 9); gamma.seed(S, 0, .01f, 1.f);
        const auto run = [&](int n, float* checkpoint = nullptr, float* history = nullptr, int after = 0) {
            strata::prefill::gdn_conv(conv.f(), raw.f(), weights.f(), h.f(), n, 1e-6f, stream, history, after);
            strata::prefill::gdn_recurrence(state.f(), h.f(), gates.f(), beta.f(), z.f(), gamma.f(), 1e-6f,
                                          y.f(), y16.h(), n, stream, checkpoint, after);
            need(cudaStreamSynchronize(stream) == cudaSuccess, "export kernel completion");
        };
        state.seed(state_bytes / 4, 19); conv.seed(C * 3, 4);
        run(T);
        const auto final_state = state.read(state_bytes), final_conv = conv.read(C * 3 * 4);
        const auto output = y.read(T * V * 4), output16 = y16.read(T * V * 2);
        for (int offset : {1, 2, 7, 64, 73}) {
            state.seed(state_bytes / 4, 19); conv.seed(C * 3, 4);
            run(T, saved.f(), conv_saved.f(), offset);
            need(state.read(state_bytes) == final_state && conv.read(C * 3 * 4) == final_conv,
                 "export changed live GDN state");
            need(y.read(T * V * 4) == output && y16.read(T * V * 2) == output16,
                 "export changed full-call GDN outputs");
            const auto exported = saved.read(state_bytes), history = conv_saved.read(C * 3 * 4);
            state.seed(state_bytes / 4, 19); conv.seed(C * 3, 4);
            run(offset);
            need(exported == state.read(state_bytes) && history == conv.read(C * 3 * 4),
                 "exported state differs from native prefix");
        }
        // Tail exports preserve the append kernel's FP16 rounding and physical
        // tail-slot convention even when the offset wraps a block.
        {
            constexpr int D = 128;
            Buffer keys(T * D * 4), tail(3 * D * 4), tail_saved(3 * D * 4), dead(D * 4), pool(32 * D * 4), block(4), norm(D * 4);
            keys.seed(T * D, 1); norm.seed(D, 3, .01f, 1.f);
            const auto shape = strata::kernels::qsa_real_shapes();
            for (int start : {0, 3, 7}) for (int offset : {1, 2, 7, 64, 73}) {
                tail.seed(3 * D, 14);
                strata::kernels::native_qsa_indexer_export_tail(keys.f(), offset, start, tail.f(), tail_saved.f(), stream);
                const strata::kernels::QsaIndexerBuffers buffers{tail.f(), dead.f(), pool.f(), static_cast<int32_t*>(block.data)};
                strata::kernels::native_qsa_indexer_append_batch(keys.f(), offset, start, 0, norm.f(), 1e-6f, buffers, shape, 120,
                                                               1e7f, stream);
                need(cudaStreamSynchronize(stream) == cudaSuccess, "indexer completion");
                need(tail_saved.read(3 * D * 4) == tail.read(3 * D * 4), "indexer tail checkpoint mismatch");
            }
        }
        {
            constexpr int P = 19, D = 10240, N = 2560, HISTORY = 9;
            Buffer key(P * D * 4), hidden(P * D * 4), value(P * N * 4), history(HISTORY * D * 4);
            Buffer norm(D * 4), conv_weights(D * 4 * 2), normalized(P * D * 4), gated(P * D * 4), gate(P * 4 * 4);
            Buffer checkpoint(HISTORY * D * 4);
            value.seed(P * N, 11); norm.seed(D, 4, .01f, 1.f);
            std::vector<uint16_t> taps(D * 4, 0x3400);  // exactly 0.25 in F16
            need(cudaMemcpy(conv_weights.data, taps.data(), taps.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess, "PLE conv weights");
            strata::kernels::PleWeights weights;
            weights.norm_key = weights.norm_query = weights.norm_conv = norm.f();
            weights.conv1d_f16 = conv_weights.h();
            const auto ple = [&](int n, int offset = 0) {
                key.seed(P * D, 14); hidden.seed(P * D, 5); history.seed(HISTORY * D, 16);
                strata::kernels::native_ple_postops_batch(key.f(), hidden.f(), value.f(), history.f(), weights,
                    normalized.f(), gated.f(), gate.f(), n, stream, offset ? checkpoint.f() : nullptr, offset);
                need(cudaStreamSynchronize(stream) == cudaSuccess, "PLE checkpoint completion");
            };
            ple(P);
            const auto output = hidden.read(P * D * 4), final = history.read(HISTORY * D * 4);
            for (int offset : {1, 7, 9, 17, 19}) {
                ple(P, offset);
                need(hidden.read(P * D * 4) == output && history.read(HISTORY * D * 4) == final,
                     "PLE export changed full-call output or live history");
                const auto saved = checkpoint.read(HISTORY * D * 4);
                ple(offset);
                need(saved == history.read(HISTORY * D * 4), "PLE exported history differs from native prefix");
            }
        }
        cudaStreamDestroy(stream);
        std::puts("GDN export: full-call bitwise parity, exact recurrence/conv prefixes, PLE history and indexer tails passed");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
