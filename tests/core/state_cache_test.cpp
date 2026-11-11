#include "strata/core/state_cache.hpp"

#include <chrono>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <cstring>

namespace sc = strata::core::statecache;
using sc::Bytes;

void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
sc::Prefix prefix(std::initializer_list<int32_t> tokens, std::string identity = "model-a") {
    return {std::move(identity), tokens, {}};
}
sc::Capture capture(sc::Prefix p, Bytes bytes, bool ready = false) {
    return {std::move(p), {{"kv", std::move(bytes)}}, ready, false};
}
sc::Config config() {
    sc::Config c; c.chunk_bytes = 4; c.ram_bytes = 128; c.disk_bytes = 128;
    c.leaf_last = false; return c;
}
class FailingStorage final : public sc::Storage {
public:
    std::atomic<int> writes{0};
    bool put(const std::string&, const Bytes&, std::string&) override { ++writes; throw std::runtime_error("disk full"); }
    bool get(const std::string&, Bytes&, std::string&) override { return false; }
    bool erase(const std::string&, std::string&) override { return true; }
    std::vector<std::string> manifests(std::string&) override { return {}; }
};

class BatchStorage final : public sc::Storage {
    std::shared_ptr<sc::Storage> local_;
public:
    size_t batches = 0, largest = 0;
    std::atomic<int> invalidations{0};
    explicit BatchStorage(std::shared_ptr<sc::Storage> local) : local_(std::move(local)) {}
    bool shared() const override { return true; }
    bool put(const std::string& k, const Bytes& b, std::string& e) override { return local_->put(k,b,e); }
    bool get(const std::string& k, Bytes& b, std::string& e) override { return local_->get(k,b,e); }
    bool erase(const std::string& k, std::string& e) override { return local_->erase(k,e); }
    std::vector<std::string> manifests(std::string& e) override { return local_->manifests(e); }
    bool get_chunks(const std::string& unit, const std::vector<std::string>& keys, std::vector<Bytes>& b, std::string& e) override {
        ++batches; largest = std::max(largest, keys.size());
        return sc::Storage::get_chunks(unit, keys, b, e);
    }
    void invalidate(const std::string& k) override { std::string e; local_->erase(k,e); ++invalidations; }
};

// Shared object ownership belongs to the producer. Consumers may retain local
// bytes, but a removed manifest must still revoke their future recovery hits.
class SharedObjects final : public sc::Storage {
    std::map<std::string, Bytes> objects_;
public:
    bool shared() const override { return true; }
    bool has_local_tier() const override { return false; }
    uint64_t local_bytes(const std::string&, uint64_t) const override { return 0; }
    bool put(const std::string& key, const Bytes& bytes, std::string&) override { objects_[key] = bytes; return true; }
    bool get(const std::string& key, Bytes& bytes, std::string&) override {
        const auto found = objects_.find(key);
        if (found == objects_.end()) return false;
        bytes = found->second; return true;
    }
    bool erase(const std::string&, std::string&) override { return true; }
    std::vector<std::string> manifests(std::string&) override { return {}; }
    std::vector<std::string> lookup(const sc::Prefix&, std::string&) override {
        std::vector<std::string> keys;
        for (const auto& [key, bytes] : objects_) { (void) bytes; if (key.starts_with("unit-")) keys.push_back(key); }
        return keys;
    }
    void invalidate(const std::string& key) override { objects_.erase(key); }
};

int main() {
    try {
        std::string error;
        check(sc::content_digest("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty SHA256");
        check(sc::content_digest("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA256 block padding");
        const std::string digest_fixture(1000, 'a');
        check(sc::content_digest(digest_fixture.data(), digest_fixture.size()) == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3", "SHA256 multiple blocks");
        {
            sc::StateCache cache(config());
            check(cache.publish(capture(prefix({1, 2}), {1, 2, 3, 4}), error), "publish prefix");
            check(cache.publish(capture(prefix({1, 2}), {1, 2, 3, 4}), error) && cache.stats().units == 1,
                  "duplicate complete publication keeps one unit");
            check(!cache.find_resume(prefix({1, 2})), "identical prompt must leave a token for logits");
            auto lease = cache.find_resume(prefix({1, 2, 3}));
            check(lease && lease.tokens() == 2, "restore deepest complete prefix");
            std::vector<sc::Region> restored;
            check(cache.restore(lease, restored, error) && restored.size() == 1 && restored[0].bytes == Bytes({1, 2, 3, 4}), "restore real payload");
            cache.release(lease);
            check(!cache.find_resume(prefix({1, 9, 3})), "different prefix never hits");
            check(!cache.find_resume(prefix({1, 2, 3}, "model-b")), "different model never hits");
            check(cache.publish(capture(prefix({1, 2}), {9, 8, 7, 6}, true), error), "publish logits-ready endpoint");
            check(bool(cache.find_resume(prefix({1, 2}))), "full prompt may hit with explicitly ready logits");
        }
        {
            sc::StateCache cache(config());
            auto p = prefix({1, 2}); p.extra = {{0, 1, "image-a"}};
            check(cache.publish(capture(p, {1, 2, 3, 4}), error), "publish multimodal state");
            auto query = p; query.tokens.push_back(3);
            check(bool(cache.find_resume(query)), "same image matches");
            query.extra[0].identity = "image-b";
            check(!cache.find_resume(query), "different image refuses reused tokens");
        }
        {
            sc::StateCache cache(config());
            check(cache.publish(capture(prefix({1}), {1, 2, 3, 4}), error), "publish shared root");
            check(cache.publish(capture(prefix({1, 2}), {1, 2, 3, 4, 5, 6, 7, 8}), error), "publish child");
            const auto stats = cache.stats();
            check(stats.logical_bytes == 12 && stats.ram_bytes == 8, "shared chunks consume physical bytes once");
        }
        {
            auto c = config(); c.ram_bytes = 8; c.chunk_bytes = 8;
            sc::StateCache cache(c);
            check(cache.publish(capture(prefix({1}), Bytes(8, 1)), error), "publish cheap state");
            check(cache.publish(capture(prefix({2, 3}), Bytes(8, 2)), error), "admit higher cost density");
            check(!cache.find_resume(prefix({1, 9})), "evict lower density state");
            auto lease = cache.find_resume(prefix({2, 3, 9}));
            check(bool(lease), "higher density retained");
            check(!cache.publish(capture(prefix({7}), Bytes(8, 3)), error), "active reader pins its entire recovery unit");
            std::vector<sc::Region> out;
            check(cache.restore(lease, out, error) && out[0].bytes == Bytes(8, 2), "pinned payload remains intact");
            cache.release(lease);
            check(cache.stats().ram_bytes <= 8 && std::isfinite(cache.stats().watermark), "bounded bytes and finite clock");
        }
        {
            const auto dir = std::filesystem::temp_directory_path() /
                ("maya-state-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            auto c = config(); c.directory = dir.string();
            {
                sc::StateCache cache(c);
                check(cache.publish(capture(prefix({4, 5}), {8, 7, 6, 5, 4, 3, 2, 1}), error), "publish persistent state");
                cache.flush();
                check(cache.stats().disk_bytes == 8 && cache.stats().write_failures == 0, "write-through finished");
                bool locked = false;
                try { sc::StateCache other(c); } catch (const std::exception&) { locked = true; }
                check(locked, "a second engine cannot race local recovery and orphan cleanup");
            }
            std::ofstream(dir / "chunk-interrupted") << "partial";
            std::ofstream(dir / "unit-interrupted.state.tmp") << "partial";
            {
                sc::StateCache cache(c);
                check(!std::filesystem::exists(dir / "chunk-interrupted") &&
                      !std::filesystem::exists(dir / "unit-interrupted.state.tmp"), "interrupted publication orphans removed at restart");
                auto lease = cache.find_resume(prefix({4, 5, 6}));
                Bytes native(8);
                sc::RegionTarget target{"kv", native.size(), [&](uint64_t at, const uint8_t* data, size_t n, std::string&) {
                    std::memcpy(native.data()+at, data, n); return true;
                }};
                check(cache.restore(lease, {target}, error) && native == Bytes({8,7,6,5,4,3,2,1}) && cache.stats().promoted == 1,
                      "bounded native restore promotes a fitting SSD unit back to RAM");
                std::vector<sc::Region> out;
                check(lease && cache.restore(lease, out, error) && out[0].bytes == Bytes({8, 7, 6, 5, 4, 3, 2, 1}), "restart restores a complete state");
                cache.release(lease);
                for (const auto& f : std::filesystem::directory_iterator(dir)) {
                    if (f.path().filename().string().starts_with("chunk-")) { std::ofstream(f.path(), std::ios::binary) << "bad"; break; }
                }
                lease = cache.find_resume(prefix({4, 5, 6}));
                check(cache.restore(lease, out, error), "resident chunks remain reusable without a disk read");
                cache.release(lease);
                check(cache.stats().promoted == 1, "complete restored unit promoted into bounded RAM");
            }
            {
                sc::StateCache cache(c);
                auto lease = cache.find_resume(prefix({4, 5, 6}));
                std::vector<sc::Region> out;
                check(!cache.restore(lease, out, error) && out.empty(), "corrupt chunk fails whole load without partial payload");
                cache.release(lease);
                check(!cache.find_resume(prefix({4, 5, 6})), "corrupt unit is not reusable");
            }
            std::filesystem::remove_all(dir);
        }
        {
            auto storage = std::make_shared<FailingStorage>();
            sc::StateCache cache(config(), storage);
            check(cache.publish(capture(prefix({7}), {1, 2, 3, 4}), error), "RAM admission before asynchronous storage failure");
            cache.flush();
            check(storage->writes > 0 && cache.stats().write_failures > 0, "worker reports exceptions and completes its job");
            auto lease = cache.find_resume(prefix({7, 8}));
            std::vector<sc::Region> out;
            check(cache.restore(lease, out, error) && out[0].bytes == Bytes({1, 2, 3, 4}), "storage failure preserves complete RAM state");
            cache.release(lease);
        }
        {
            sc::StateCache cache(config());
            cache.observe_prefill(0, 100, 3);
            cache.observe_prefill(100, 200, 7);
            const auto stats = cache.stats();
            check(std::abs(stats.cost_per_token - .01) < 1e-12 && std::abs(stats.cost_per_token_per_100k - 40) < 1e-8,
                  "runtime calibration fits actual linear and attention terms");
            auto state = capture(prefix({1}), {1, 2, 3, 4});
            state.min_prompt_tokens = 4;
            check(cache.publish(std::move(state), error), "publish window-qualified checkpoint");
            check(!cache.find_resume(prefix({1, 2, 3})), "shorter prompt cannot reuse missing draft context");
            auto lease = cache.find_resume(prefix({1, 2, 3, 4}));
            check(bool(lease), "qualified prompt matches");
            cache.invalidate(prefix({1, 2, 3}));
            std::vector<sc::Region> out;
            check(!cache.restore(lease, out, error), "invalidated unit cannot be read through an old lease");
            check(cache.stats().units == 1, "invalidated reader pins storage through completion");
            cache.release(lease);
            check(cache.stats().units == 0 && cache.stats().ram_bytes == 0, "dead unit and its chunks reclaimed on release");
        }
        {
            const std::vector<uint64_t> marks = {8, 16};
            auto p = sc::plan_export(0, 24, marks, 8, true);
            check(p.end == 16 && p.checkpoint == 8, "one in-chunk export before next required mark");
            p = sc::plan_export(0, 24, marks, 16, true);
            check(p.end == 8 && p.checkpoint == 0, "unsupported offset cuts exactly");
            p = sc::plan_export(0, 24, marks, 1, false);
            check(p.end == 8 && p.checkpoint == 0, "backend fallback is explicit");
            uint64_t endpoint = 0;
            check(sc::Commit{10, 8, 3, 3, true}.exact(endpoint) && endpoint == 13, "accepted state endpoint");
            check(!sc::Commit{10, 8, 3, 2, true}.exact(endpoint), "stop inside accepted window must resolve before publication");
            check(!sc::Commit{10, 8, 3, 3, false}.exact(endpoint), "in-flight state is not exact");
        }
        {
            auto c = config(); c.supersede_lag = 1;
            sc::StateCache cache(c);
            check(cache.publish(capture(prefix({1,2}), Bytes(4,1)), error), "interior fallback");
            for (int n = 3; n <= 5; ++n) {
                auto state = capture(prefix({1,2,3,4,5}), Bytes(4, uint8_t(n)));
                state.prefix.tokens.resize(size_t(n)); state.frontier = true;
                check(cache.publish(std::move(state), error), "complete frontier chain");
            }
            auto lease = cache.find_resume(prefix({1,2,3,9}));
            check(lease && lease.tokens() == 2, "supersession retires old frontier and preserves interior fallback");
            cache.release(lease);
            lease = cache.find_resume(prefix({1,2,3,4,9}));
            check(lease && lease.tokens() == 4, "lag retains previous complete answer for rewrites");
            cache.release(lease);
        }
        {
            auto c = config(); c.supersede_lag = 0;
            sc::StateCache cache(c);
            auto root = capture(prefix({1,2}), Bytes(4,1)); root.frontier = true;
            check(cache.publish(std::move(root), error), "branch root");
            check(cache.publish(capture(prefix({1,2,9}), Bytes(4,2)), error), "sibling continuation");
            auto end = capture(prefix({1,2,3,4}), Bytes(4,3)); end.frontier = true;
            check(cache.publish(std::move(end), error), "other branch frontier");
            auto lease = cache.find_resume(prefix({1,2,8}));
            check(lease && lease.tokens() == 2, "shared branch point survives frontier supersession");
            cache.release(lease);
        }
        {
            const auto dir = std::filesystem::temp_directory_path() /
                ("maya-state-stream-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            auto c = config(); c.directory = dir.string(); c.ram_bytes = 8; c.disk_bytes = 128;
            Bytes native(64);
            for (size_t i = 0; i < native.size(); ++i) native[i] = uint8_t(i);
            sc::RegionSource source{"kv", native.size(), [&](uint64_t at, uint8_t* out, size_t n, std::string&) {
                check(n <= c.chunk_bytes && at+n <= native.size(), "capture staging remains bounded");
                std::memcpy(out, native.data()+at, n); return true;
            }};
            {
                sc::StateCache cache(c);
                sc::Capture metadata; metadata.prefix = prefix({1,2});
                check(cache.publish(std::move(metadata), {source}, error), "capture larger than RAM directly into SSD");
                cache.flush();
                check(cache.stats().ram_bytes == 0 && cache.stats().disk_bytes == native.size(), "large state never materializes in RAM tier");
            }
            {
                auto storage = std::make_shared<BatchStorage>(sc::file_storage(dir.string()));
                sc::StateCache cache(c, storage);
                auto lease = cache.find_resume(prefix({1,2,3}));
                check(bool(lease), "large persisted unit recovered after restart");
                Bytes restored(native.size(), 0);
                size_t writes = 0;
                sc::RegionTarget target{"kv", native.size(), [&](uint64_t at, const uint8_t* data, size_t n, std::string&) {
                    check(n <= c.chunk_bytes && at+n <= restored.size(), "restore staging remains bounded");
                    ++writes; std::memcpy(restored.data()+at, data, n); return true;
                }};
                auto broken = target; ++broken.bytes;
                check(!cache.restore(lease, {broken}, error) && writes == 0, "large restore validates all layouts before writes");
                check(cache.restore(lease, {target}, error) && restored == native, "large SSD unit restores with bounded native writes");
                cache.restored(lease); cache.restored(lease);
                check(cache.stats().restored_tokens == 2, "native completion acknowledged once per lease");
                cache.release(lease);
                check(cache.stats().ram_bytes == 0, "large restore does not promote a full unit beyond host cap");
                check(storage->batches == 1 && storage->largest == 16, "native restore batch-loads absent chunks");
                cache.invalidate(prefix({1,2})); cache.flush();
                check(storage->invalidations == 1 && !cache.find_resume(prefix({1,2,3})), "explicit invalidation revokes shared recovery metadata");
            }
            std::filesystem::remove_all(dir);
        }
        {
            const auto dir = std::filesystem::temp_directory_path() /
                ("maya-state-stream-fail-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            auto c = config(); c.directory = dir.string(); c.ram_bytes = 4;
            {
                sc::StateCache cache(c);
                size_t reads = 0;
                sc::RegionSource source{"kv", 32, [&](uint64_t at, uint8_t* out, size_t n, std::string& e) {
                    if (++reads == 11) { e = "injected native copy failure"; return false; }
                    for (size_t i = 0; i < n; ++i) out[i] = uint8_t(at+i);
                    return true;
                }};
                sc::Capture metadata; metadata.prefix = prefix({1,2});
                check(!cache.publish(std::move(metadata), {source}, error), "failed streaming copy is rejected");
                check(cache.stats().units == 0 && cache.stats().disk_bytes == 0 && !cache.find_resume(prefix({1,2,3})),
                      "partial streamed payload never publishes or leaks its tier reservation");
            }
            for (const auto& file : std::filesystem::directory_iterator(dir))
                check(file.path().filename() == "owner.lock", "failed stream reclaims unpublished local chunks");
            std::filesystem::remove_all(dir);
        }
        {
            auto c = config(); c.disk_bytes = 0;
            auto storage = std::make_shared<SharedObjects>();
            sc::StateCache producer(c, storage), consumer(c, storage);
            const auto state = capture(prefix({1,2}), {1,2,3,4});
            check(producer.publish(state, error), "shared producer capture"); producer.flush();
            auto lease = consumer.find_resume(prefix({1,2,3}));
            std::vector<sc::Region> out;
            check(lease && consumer.restore(lease, out, error) && out[0].bytes == Bytes({1,2,3,4}), "consumer imports shared bytes");
            consumer.release(lease); consumer.flush();
            check(consumer.stats().ram_bytes == 4, "imported state promoted into consumer RAM");
            producer.invalidate(state.prefix); producer.flush();
            check(!consumer.find_resume(prefix({1,2,3})), "revoked shared manifest cannot resume from cached RAM");
        }
        std::cout << "state cache: all behavioral checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
