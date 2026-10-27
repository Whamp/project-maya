// Expert read-ahead policy. Configuration is immutable after its first use in a process.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::ds4 {

enum class ExpertPrefetchMode { Sampled, Always };

/// Expert prefetch sample counts are page probes, not a count of pages omitted from a hint.
struct ExpertPrefetchPolicy {
    ExpertPrefetchMode mode;
    int samples;
    const char* mode_source;
    const char* samples_source;
};

/// Parse expert prefetch settings without reading the environment; invalid explicit settings throw.
inline ExpertPrefetchPolicy parse_expert_prefetch_policy(const char* mode, const char* samples, const char* legacy_all) {
#if defined(_WIN32)
    constexpr auto default_mode = ExpertPrefetchMode::Sampled;
    constexpr int default_samples = 8;
#else
    constexpr auto default_mode = ExpertPrefetchMode::Always;
    constexpr int default_samples = 4;
#endif
    ExpertPrefetchPolicy policy{default_mode, default_samples, "default", "default"};
    if (mode != nullptr) {
        if (std::strcmp(mode, "sampled") == 0) policy.mode = ExpertPrefetchMode::Sampled;
        else if (std::strcmp(mode, "always") == 0) policy.mode = ExpertPrefetchMode::Always;
        else throw std::invalid_argument("DS4_EXPERT_PREFETCH_MODE: expected sampled or always");
        policy.mode_source = "DS4_EXPERT_PREFETCH_MODE";
    } else if (legacy_all != nullptr) {
        // Preserve the platform default for zero, including Linux\'s always-hint policy.
        policy.mode = std::atoi(legacy_all) != 0 ? ExpertPrefetchMode::Always : default_mode;
        policy.mode_source = "DS4_PF_ALL";
    }
    if (samples != nullptr) {
        int count = 0;
        for (const char* at = samples; *at != '\0'; ++at) {
            if (*at < '0' || *at > '9')
                throw std::invalid_argument("DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16");
            count = count * 10 + (*at - '0');
            if (count > 16)
                throw std::invalid_argument("DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16");
        }
        if (count == 0)
            throw std::invalid_argument("DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16");
        policy.samples = count;
        policy.samples_source = "DS4_EXPERT_PREFETCH_SAMPLES";
    }
    return policy;
}

/// Read and log the effective expert prefetch policy once, before model loading in maya-ds4.
inline const ExpertPrefetchPolicy& expert_prefetch_policy() {
    static const ExpertPrefetchPolicy policy = [] {
        const auto value = parse_expert_prefetch_policy(std::getenv("DS4_EXPERT_PREFETCH_MODE"),
                                                       std::getenv("DS4_EXPERT_PREFETCH_SAMPLES"),
                                                       std::getenv("DS4_PF_ALL"));
        std::fprintf(stderr, "ds4: expert prefetch mode=%s source=%s samples=%d samples_source=%s%s\n",
                     value.mode == ExpertPrefetchMode::Always ? "always" : "sampled", value.mode_source,
                     value.samples, value.samples_source,
                     value.mode == ExpertPrefetchMode::Always ? " (samples inactive)" : "");
        return value;
    }();
    return policy;
}

/// The separate IQ prefetch override bypasses the shared policy; log it only when the IQ path uses it.
inline bool iq_expert_prefetch_always() {
    static const bool always = [] {
        const char* value = std::getenv("DS4_IQ_PF_ALL");
        const bool enabled = value != nullptr && std::atoi(value) != 0;
        if (enabled)
            std::fprintf(stderr, "ds4: DS4_IQ_PF_ALL bypasses the shared expert prefetch policy for IQ ranges\n");
        return enabled;
    }();
    return always;
}

}  // namespace strata::ds4
