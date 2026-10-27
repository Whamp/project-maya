// Fresh processes exercise cached environment parsing and once-only policy/override logs.
#include "../src/ds4/ds4_expert_prefetch.hpp"

#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {
struct PolicyCase {
    const char* mode;
    const char* samples;
    const char* legacy;
    const char* iq;
    int exit_code;
    const char* expected;
};

void set_policy_variable(const char* name, const char* value) {
    if (value != nullptr) setenv(name, value, 1);
    else unsetenv(name);
}

bool check_policy_process(const char* executable, const PolicyCase& test) {
    int output[2];
    if (pipe(output) != 0) return false;
    const pid_t child = fork();
    if (child == 0) {
        close(output[0]);
        if (dup2(output[1], STDERR_FILENO) < 0) _exit(125);
        close(output[1]);
        set_policy_variable("DS4_EXPERT_PREFETCH_MODE", test.mode);
        set_policy_variable("DS4_EXPERT_PREFETCH_SAMPLES", test.samples);
        set_policy_variable("DS4_PF_ALL", test.legacy);
        set_policy_variable("DS4_IQ_PF_ALL", test.iq);
        execl(executable, executable, "--probe", nullptr);
        _exit(125);
    }
    close(output[1]);
    std::string log;
    char buffer[1024];
    ssize_t bytes;
    while ((bytes = read(output[0], buffer, sizeof buffer)) > 0) log.append(buffer, static_cast<size_t>(bytes));
    close(output[0]);
    if (child < 0) return false;
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status)) return false;
    bool ok = WEXITSTATUS(status) == test.exit_code && log.find(test.expected) != std::string::npos;
    const std::string policy_prefix = "ds4: expert prefetch mode=";
    const size_t first = log.find(policy_prefix);
    if (test.exit_code == 0) {
        ok = ok && first != std::string::npos && log.find(policy_prefix, first + 1) == std::string::npos;
        const std::string iq_prefix = "ds4: DS4_IQ_PF_ALL bypasses";
        const size_t iq_at = log.find(iq_prefix);
        const bool iq_enabled = test.iq != nullptr && std::atoi(test.iq) != 0;
        ok = ok && (iq_enabled ? iq_at != std::string::npos && log.find(iq_prefix, iq_at + 1) == std::string::npos
                               : iq_at == std::string::npos);
    } else ok = ok && first == std::string::npos;
    if (!ok) std::fprintf(stderr, "ds4 prefetch policy test: expected exit %d and '%s', got exit %d: %s\n",
                          test.exit_code, test.expected, WEXITSTATUS(status), log.c_str());
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--probe") == 0) {
        try {
            const auto& first = strata::ds4::expert_prefetch_policy();
            const int samples = first.samples;
            setenv("DS4_EXPERT_PREFETCH_MODE", "invalid-after-first-use", 1);
            setenv("DS4_EXPERT_PREFETCH_SAMPLES", "0", 1);
            if (&first != &strata::ds4::expert_prefetch_policy() || first.samples != samples) return 3;
            (void) strata::ds4::iq_expert_prefetch_always();
            (void) strata::ds4::iq_expert_prefetch_always();
            return 0;
        } catch (const std::invalid_argument& error) {
            std::fprintf(stderr, "%s\n", error.what());
            return 2;
        }
    }
    const PolicyCase cases[] = {
#if defined(_WIN32)
        {nullptr, nullptr, nullptr, nullptr, 0, "mode=sampled source=default samples=8 samples_source=default"},
        {nullptr, nullptr, "0", nullptr, 0, "mode=sampled source=DS4_PF_ALL"},
        {nullptr, nullptr, "garbage", nullptr, 0, "mode=sampled source=DS4_PF_ALL"},
#else
        {nullptr, nullptr, nullptr, nullptr, 0, "mode=always source=default samples=4 samples_source=default"},
        {nullptr, "8", nullptr, nullptr, 0, "mode=always source=default samples=8 samples_source=DS4_EXPERT_PREFETCH_SAMPLES"},
        {nullptr, nullptr, "0", nullptr, 0, "mode=always source=DS4_PF_ALL"},
        {nullptr, nullptr, "garbage", nullptr, 0, "mode=always source=DS4_PF_ALL"},
#endif
        {nullptr, nullptr, "1", nullptr, 0, "mode=always source=DS4_PF_ALL"},
        {nullptr, nullptr, "-1", nullptr, 0, "mode=always source=DS4_PF_ALL"},
        {"sampled", "1", "1", nullptr, 0, "mode=sampled source=DS4_EXPERT_PREFETCH_MODE samples=1"},
        {"always", "16", "0", nullptr, 0, "mode=always source=DS4_EXPERT_PREFETCH_MODE samples=16"},
        {"sampled", "2", nullptr, nullptr, 0, "samples=2 samples_source=DS4_EXPERT_PREFETCH_SAMPLES"},
        {"sampled", "4", nullptr, nullptr, 0, "samples=4 samples_source=DS4_EXPERT_PREFETCH_SAMPLES"},
        {"sampled", "8", nullptr, "1", 0, "samples=8 samples_source=DS4_EXPERT_PREFETCH_SAMPLES"},
        {"sampled", "16", nullptr, "-1", 0, "samples=16 samples_source=DS4_EXPERT_PREFETCH_SAMPLES"},
        {"always", nullptr, nullptr, "0", 0, "(samples inactive)"},
        {"", nullptr, "1", nullptr, 2, "DS4_EXPERT_PREFETCH_MODE: expected sampled or always"},
        {"SAMPLED", nullptr, nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_MODE: expected sampled or always"},
        {" sampled", nullptr, nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_MODE: expected sampled or always"},
        {nullptr, "", nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16"},
        {nullptr, "0", nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16"},
        {nullptr, "17", nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16"},
        {nullptr, "-1", nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16"},
        {nullptr, "+4", nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16"},
        {nullptr, " 4", nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16"},
        {nullptr, "4x", nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16"},
        {nullptr, "4.0", nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16"},
        {nullptr, "4 ", nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16"},
        {nullptr, "99999999999999999999", nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16"},
        {"always", "17", nullptr, nullptr, 2, "DS4_EXPERT_PREFETCH_SAMPLES: expected an integer from 1 through 16"},
    };
    bool ok = true;
    for (const auto& test : cases) ok = check_policy_process(argv[0], test) && ok;
    if (ok) std::puts("ds4_prefetch_policy_test: all fresh-process checks passed");
    return ok ? 0 : 1;
}
