#include "glm_metal.hpp"
#include "mm3_metal.hpp"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>

namespace sc = strata::core::statecache;
void need(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
template <class Model> void run(Model& model, const std::string& file) {
    std::string error;
    need(model.load(file, 128, error), error);
    sc::Prefix prefix{model.state_identity(), {1,2,3,4,5,6,7,8,9}, {}};
    std::vector<float> baseline, restored;
    need(model.forward(prefix.tokens, baseline, error), error);
    sc::Capture state;
    need(model.state_capture(state, prefix, error), error);
    need(model.forward({10}, baseline, error), error);
    model.reset();
    need(model.forward({17,18,19,20,21,22,23,24,25,26,27,28}, restored, error), error);
    // Truncation must fail before writing any tensor or changing the endpoint.
    auto broken = state.regions; broken.pop_back();
    const auto before = model.position();
    need(!model.state_restore(prefix, broken, error) && model.position() == before, "partial unit changed live state");
    error.clear();
    need(model.state_restore(prefix, state.regions, error), error);
    need(model.forward({10}, restored, error), error);
    need(restored.size() == baseline.size(), "vocabulary mismatch");
    for (size_t i = 0; i < restored.size(); ++i)
        need(std::isfinite(restored[i]) && std::abs(restored[i]-baseline[i]) < 1e-5f, "restored logits changed");
}
int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: metal_state_cache_test GLM_GGUF MM3_GGUF\n"); return 2; }
    try {
        maya::metal::GlmMetal glm; maya::metal::GlmOptions o;
        o.use_gpu = false; o.cpu_threads = 2; o.pool_slots = 8; o.n_ubatch = 16; o.max_window = 8; o.mtp = true;
        glm.set_options(o); run(glm, argv[1]);
        maya::metal::Mm3Metal mm3; maya::metal::Mm3Options m;
        m.use_gpu = false; m.cpu_threads = 2; m.n_ubatch = 16;
        mm3.set_options(m); run(mm3, argv[2]);
        std::puts("Metal graph state: GLM and MiniMax complete restores, overwritten context and malformed-unit rejection passed");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
