#include "strata/core/state_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <span>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace strata::core::statecache {
namespace {
struct Elapsed {   // adds the wall time of a scope to a counter
    double& total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~Elapsed() { total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
};
uint64_t hash(std::span<const uint8_t> b) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t v : b) h = (h ^ v) * 1099511628211ull;
    return h;
}
double number(const char* key, double fallback) {
    const char* value = std::getenv(key);
    if (!value) return fallback;
    char* end = nullptr;
    const double n = std::strtod(value, &end);
    return end != value && *end == 0 && std::isfinite(n) && n >= 0 ? n : fallback;
}
uint64_t capacity(const char* key, uint64_t fallback, double scale = 1) {
    const double value = number(key, double(fallback) / scale) * scale;
    // Conversion of an out-of-range floating value to an integer is undefined.
    if (value >= double(std::numeric_limits<uint64_t>::max())) return fallback;
    return uint64_t(value);
}
void put(Bytes& b, uint64_t n) {
    for (int i = 0; i < 8; ++i) b.push_back(uint8_t(n >> (i * 8)));
}
void put(Bytes& b, const std::string& s) {
    put(b, s.size()); b.insert(b.end(), s.begin(), s.end());
}
struct Reader {
    const Bytes& b;
    size_t at = 0;
    uint64_t n() {
        if (b.size() - at < 8) throw std::runtime_error("truncated state manifest");
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= uint64_t(b[at++]) << (i * 8);
        return v;
    }
    size_t count(size_t minimum_bytes = 1) {
        const uint64_t v = n();
        if (v > (b.size() - at) / minimum_bytes) throw std::runtime_error("invalid state manifest count");
        return size_t(v);
    }
    std::string s() {
        const size_t len = count();
        std::string v(reinterpret_cast<const char*>(b.data() + at), len); at += len; return v;
    }
};
class Files final : public Storage {
    std::filesystem::path root_;
    mutable std::mutex sizes_mutex_;
    std::mutex put_mutex_;
    std::map<std::string, uint64_t> sizes_;
#ifdef _WIN32
    HANDLE owner_ = INVALID_HANDLE_VALUE;
#else
    int owner_ = -1;
#endif
    bool valid(const std::string& key) const {
        return !key.empty() && key.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-.") == std::string::npos;
    }
public:
    explicit Files(std::string path) : root_(std::move(path)) {
        std::filesystem::create_directories(root_);
        const auto lock = root_ / "owner.lock";
#ifdef _WIN32
        owner_ = CreateFileW(lock.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (owner_ == INVALID_HANDLE_VALUE) throw std::runtime_error("state directory is owned by another engine");
#else
        owner_ = ::open(lock.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (owner_ < 0 || flock(owner_, LOCK_EX | LOCK_NB) != 0) {
            if (owner_ >= 0) ::close(owner_);
            throw std::runtime_error("state directory is unavailable or owned by another engine");
        }
#endif
        std::error_code ec;
        for (const auto& f : std::filesystem::directory_iterator(root_, ec))
            if (f.is_regular_file(ec)) sizes_[f.path().filename().string()] = f.file_size(ec);
    }
    ~Files() override {
#ifdef _WIN32
        if (owner_ != INVALID_HANDLE_VALUE) CloseHandle(owner_);
#else
        if (owner_ >= 0) ::close(owner_);
#endif
    }
    bool put(const std::string& key, const Bytes& data, std::string& error) override {
        std::lock_guard serialization(put_mutex_);
        if (!valid(key)) { error = "invalid state storage key"; return false; }
        const auto path = root_ / key;
        const auto tmp = root_ / (key + ".tmp");
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
        f.close();
        if (!f) {
            std::error_code cleanup; std::filesystem::remove(tmp, cleanup);
            error = "cannot write state object " + key; return false;
        }
        std::error_code ec;
#ifdef _WIN32
        if (!MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            ec = std::error_code(int(GetLastError()), std::system_category());
#else
        std::filesystem::rename(tmp, path, ec);
#endif
        if (ec) {
            error = ec.message();
            std::error_code cleanup; std::filesystem::remove(tmp, cleanup);
            return false;
        }
        { std::lock_guard lock(sizes_mutex_); sizes_[key] = data.size(); }
        return true;
    }
    bool get(const std::string& key, Bytes& data, std::string& error) override {
        if (!valid(key)) { error = "invalid state storage key"; return false; }
        std::ifstream f(root_ / key, std::ios::binary | std::ios::ate);
        const auto length = f.tellg();
        if (!f || length < 0 || uint64_t(length) > (64ull << 20)) { error = "missing or oversized state object " + key; return false; }
        try { data.resize(size_t(length)); }
        catch (const std::bad_alloc&) { error = "state object does not fit in RAM"; return false; }
        f.seekg(0);
        if (!f.read(reinterpret_cast<char*>(data.data()), length)) { error = "truncated state object " + key; return false; }
        return true;
    }
    bool erase(const std::string& key, std::string& error) override {
        if (!valid(key)) { error = "invalid state storage key"; return false; }
        std::error_code ec; std::filesystem::remove(root_ / key, ec);
        if (!ec) { std::lock_guard lock(sizes_mutex_); sizes_.erase(key); }
        if (ec) error = ec.message();
        return !ec;
    }
    std::vector<std::string> manifests(std::string& error) override {
        std::vector<std::string> result;
        std::error_code ec;
        for (const auto& f : std::filesystem::directory_iterator(root_, ec))
            if (f.path().extension() == ".state") result.push_back(f.path().filename().string());
        if (ec) error = ec.message();
        std::sort(result.begin(), result.end());
        return result;
    }
    void prune(const std::vector<std::string>& live, std::string& error) override {
        const std::set<std::string> keep(live.begin(), live.end());
        std::error_code ec;
        for (const auto& f : std::filesystem::directory_iterator(root_, ec)) {
            const auto name = f.path().filename().string();
            if ((name.starts_with("chunk-") || name.starts_with("unit-")) && !keep.count(name)) {
                std::filesystem::remove(f.path(), ec);
                if (!ec) { std::lock_guard lock(sizes_mutex_); sizes_.erase(name); }
            }
            if (ec) break;
        }
        if (ec) error = ec.message();
    }
    uint64_t local_bytes(const std::string& key, uint64_t) const override {
        std::lock_guard lock(sizes_mutex_);
        const auto it = sizes_.find(key); return it == sizes_.end() ? 0 : it->second;
    }
    double age_seconds(const std::string& key) const override {
        std::error_code ec;
        const auto written = std::filesystem::last_write_time(root_ / key, ec);
        return ec ? 0 : std::chrono::duration<double>(std::filesystem::file_time_type::clock::now()-written).count();
    }
    void touch(const std::string& key) override {
        std::error_code ec;
        std::filesystem::last_write_time(root_ / key, std::filesystem::file_time_type::clock::now(), ec);
    }
};
}

bool extends(const Prefix& a, const Prefix& b) {
    if (a.identity != b.identity || a.tokens.size() < b.tokens.size() ||
        !std::equal(b.tokens.begin(), b.tokens.end(), a.tokens.begin())) return false;
    std::vector<ExtraInput> extra;
    for (const auto& x : a.extra) if (x.position < b.tokens.size()) extra.push_back(x);
    return extra == b.extra;
}

double CostModel::seconds(uint64_t a, uint64_t b) const {
    if (b <= a) return 0;
    return double(b - a) * per_token + per_token_per_100k * double(b - a) * (double(b) + double(a)) / 200000.;
}

Config Config::environment(const std::string& dir, const std::string& family) {
    Config c;
    if (!family.empty()) {
        const auto old = "MAYA_" + family + "_";
        if (!std::getenv("MAYA_STATE_RAM_GB")) {
            const auto alias = capacity((old+"SLOT_RAM_GB").c_str(), c.ram_bytes, 1073741824.);
            if (alias) c.ram_bytes = alias;
        }
        if (!std::getenv("MAYA_STATE_DISK_GB")) c.disk_bytes = capacity((old+"SLOT_GB").c_str(), c.disk_bytes, 1073741824.);
        if (!std::getenv("MAYA_STATE_MAX")) c.max_units = size_t(capacity((old+(family == "DS4" ? "SLOT_CAP" : "SLOTS")).c_str(), c.max_units));
        c.max_age_seconds = number((old+"SLOT_DAYS").c_str(), 7) * 86400.;
        bool aliases = false;
        for (const char* key : {"SLOT_RAM_GB", "SLOT_GB", "SLOT_DAYS", family == "DS4" ? "SLOT_CAP" : "SLOTS"})
            aliases = aliases || std::getenv((old+key).c_str()) != nullptr;
        if (aliases) {
            static std::once_flag notice;
            std::call_once(notice, [] { std::fprintf(stderr, "state cache: legacy slot budgets are translated; MAYA_STATE_* settings take precedence\n"); });
        }
    }
    c.ram_bytes = capacity("MAYA_STATE_RAM_GB", c.ram_bytes, 1073741824.);
    c.disk_bytes = capacity("MAYA_STATE_DISK_GB", c.disk_bytes, 1073741824.);
    c.max_units = size_t(std::min<uint64_t>(capacity("MAYA_STATE_MAX", c.max_units), std::numeric_limits<size_t>::max()));
    c.idle_seconds = number("MAYA_STATE_IDLE_SECONDS", 60);
    c.max_age_seconds = number("MAYA_STATE_DAYS", c.max_age_seconds / 86400.) * 86400.;
    c.leaf_last = number("MAYA_STATE_LEAF_LAST", 1) != 0;
    c.anchor_tier = number("MAYA_STATE_ANCHOR_TIER", 1) != 0;
    c.write_through = number("MAYA_STATE_WRITE_THROUGH", 1) != 0;
    c.supersede_lag = size_t(capacity("MAYA_STATE_SUPERSEDE_LAG", 2));
    c.cost.per_token = number("MAYA_STATE_COST_D", 1);
    c.cost.per_token_per_100k = number("MAYA_STATE_COST_E", 0);
    c.calibrate = number("MAYA_STATE_COST_CALIBRATE", std::getenv("MAYA_STATE_COST_D") || std::getenv("MAYA_STATE_COST_E") ? 0 : 1) != 0;
    const char* path = std::getenv("MAYA_STATE_DIR"); c.directory = path ? path : dir;
    return c;
}

std::shared_ptr<Storage> file_storage(const std::string& path) { return std::make_shared<Files>(path); }
RegionSource bytes_source(std::string name, Bytes bytes) {
    auto owned = std::make_shared<const Bytes>(std::move(bytes));
    return {std::move(name), owned->size(), [owned](uint64_t at, uint8_t* data, size_t n, std::string& error) {
        if (at > owned->size() || n > owned->size()-at) { error = "state sideband range invalid"; return false; }
        std::memcpy(data, owned->data()+at, n); return true;
    }};
}
RegionTarget check_region(std::string name, Bytes bytes) {
    auto expected = std::make_shared<const Bytes>(std::move(bytes));
    return {std::move(name), expected->size(), [expected](uint64_t at, const uint8_t* data, size_t n, std::string& error) {
        if (at > expected->size() || n > expected->size()-at || std::memcmp(data, expected->data()+at, n)) {
            error = "state sideband does not match native layout"; return false;
        }
        return true;
    }};
}

struct Chunk {
    std::string key;
    uint64_t checksum = 0, size = 0;
    std::shared_ptr<const Bytes> data;
    bool stored = false;
    bool reserved_ram = false, reserved_disk = false;
};
struct Part { std::string name; std::vector<std::shared_ptr<Chunk>> chunks; };
struct StateCache::Unit {
    Prefix prefix;
    std::vector<Part> parts;
    std::string key;
    double stamp = 0, used_at = 0;
    uint64_t uses = 0;
    uint64_t min_prompt_tokens = 0;
    bool sampling_ready = false, anchor = false, frontier = false, ram = true, stored = false, pending = false, dead = false;
    bool imported = false;
};

struct StateCache::Impl {
    Config config;
    std::shared_ptr<Storage> storage;
    std::function<double()> clock;
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::map<std::string, std::shared_ptr<Unit>> units;
    std::unordered_map<uint64_t, std::vector<std::weak_ptr<Chunk>>> chunks;
    struct Job {
        std::shared_ptr<Unit> unit;
        char operation = 'W';
        std::string key;
        Job() = default;
        Job(std::shared_ptr<Unit> u) : unit(std::move(u)) {}
        Job(char op, std::string id) : operation(op), key(std::move(id)) {}
        bool payload_ready = false;
    };
    std::deque<Job> writes;
    size_t active_jobs = 0, active_updates = 0;
    Stats stats;
    double cost_xx = 0, cost_xy = 0, cost_yy = 0, cost_xt = 0, cost_yt = 0;
    bool stopping = false;
    std::thread worker;

    Impl(Config c, std::shared_ptr<Storage> s, std::function<double()> now)
        : config(std::move(c)), storage(std::move(s)), clock(std::move(now)) {
        if (!config.chunk_bytes || config.chunk_bytes > (64ull << 20)) throw std::invalid_argument("state cache chunk size must be between 1 and 64 MiB");
        stats.cost_per_token = config.cost.per_token;
        stats.cost_per_token_per_100k = config.cost.per_token_per_100k;
        if (!clock) clock = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        if (!storage && !config.directory.empty() && config.disk_bytes) storage = file_storage(config.directory);
        if (const char* backend = std::getenv("MAYA_STATE_BACKEND"); backend && std::string(backend) != "local") {
            if (std::string(backend) != "mooncake") throw std::invalid_argument("unknown MAYA_STATE_BACKEND");
#ifdef STRATA_STATE_MOONCAKE
            storage = mooncake_storage(std::move(storage));
#else
            throw std::invalid_argument("Mooncake transport is unavailable in this build; use MAYA_STATE_BACKEND=local");
#endif
        }
        if (storage) {
            recover(); trim();
            std::vector<std::string> live;
            for (const auto& [key, unit] : units) {
                live.push_back(key);
                for (const auto& part : unit->parts) for (const auto& chunk : part.chunks) live.push_back(chunk->key);
            }
            std::string error;
            storage->prune(live, error);
            if (storage->shared() && config.write_through) for (auto& [key, unit] : units) {
                (void) key;
                unit->pending = true; writes.push_back(unit);
            }
            worker = std::thread([this] { run(); });
        }
    }
    ~Impl() {
        { std::lock_guard lock(mutex); stopping = true; cv.notify_all(); }
        if (worker.joinable()) worker.join();
    }
    void control(char operation, const std::string& key) {
        if (writes.size() >= 65536) { ++stats.write_failures; return; }
        writes.emplace_back(operation, key); cv.notify_one();
    }

    std::set<const Chunk*> holders(const Unit& u) const {
        std::set<const Chunk*> set;
        for (const auto& p : u.parts) for (const auto& c : p.chunks) set.insert(c.get());
        return set;
    }
    void account() {
        stats.ram_bytes = stats.disk_bytes = stats.logical_bytes = 0;
        stats.unique_bytes = stats.shared_bytes = stats.pinned_units = 0;
        stats.pending_writes = 0;
        stats.pending_updates = active_updates;
        for (const auto& job : writes) stats.pending_updates += job.operation != 'W';
        stats.units = units.size();
        std::set<const Chunk*> seen;
        for (const auto& [key, u] : units) {
            (void) key;
            stats.pending_writes += u->pending;
            stats.pinned_units += pinned(u);
            for (const auto& p : u->parts) for (const auto& c : p.chunks) {
                stats.logical_bytes += c->size;
                if (seen.insert(c.get()).second) {
                    stats.unique_bytes += c->size;
                    if (c->data || c->reserved_ram) stats.ram_bytes += c->size;
                    const auto disk = c->stored && storage ? storage->local_bytes(c->key, c->size) : 0;
                    stats.disk_bytes += c->reserved_disk ? std::max(disk, c->size) : disk;
                }
            }
        }
        stats.shared_bytes = stats.logical_bytes - stats.unique_bytes;
    }
    uint64_t freeable(const Unit& unit, bool ram) const {
        auto alone = holders(unit);
        for (const auto& [key, other] : units) {
            (void) key;
            if (other.get() == &unit || (ram && !other->ram && other.use_count() == 1 && !other->pending)) continue;
            for (const auto& p : other->parts) for (const auto& c : p.chunks) alone.erase(c.get());
        }
        uint64_t bytes = 0;
        for (const auto* c : alone) {
            if (ram && (c->data || c->reserved_ram)) bytes += c->size;
            else if (!ram && c->reserved_disk) bytes += c->size;
            else if (!ram && c->stored && storage) bytes += storage->local_bytes(c->key, c->size);
        }
        return bytes;
    }
    double gain(const Unit& unit) const {
        if (unit.dead) return 0;
        size_t fallback = 0;
        for (const auto& [key, u] : units) {
            (void) key;
            if (u.get() != &unit && !u->dead && u->prefix.tokens.size() < unit.prefix.tokens.size() &&
                extends(unit.prefix, u->prefix)) fallback = std::max(fallback, u->prefix.tokens.size());
        }
        return config.cost.seconds(fallback, unit.prefix.tokens.size());
    }
    void retire_frontiers(const Unit& newest) {
        if (!newest.frontier) return;
        std::vector<Unit*> ancestors;
        for (auto& [key, u] : units) {
            (void) key;
            if (u.get() != &newest && u->frontier && !u->anchor && !u->dead &&
                u->prefix.tokens.size() < newest.prefix.tokens.size() && extends(newest.prefix, u->prefix))
                ancestors.push_back(u.get());
        }
        std::sort(ancestors.begin(), ancestors.end(), [](const auto* a, const auto* b) {
            return a->prefix.tokens.size() > b->prefix.tokens.size();
        });
        for (size_t i = config.supersede_lag; i < ancestors.size(); ++i) {
            Unit& old = *ancestors[i];
            bool branch = false;
            for (const auto& [key, other] : units) {
                (void) key;
                if (!other->dead && extends(other->prefix, old.prefix) &&
                    !extends(newest.prefix, other->prefix) && !extends(other->prefix, newest.prefix)) branch = true;
            }
            if (branch) continue;
            old.dead = true;
            if (storage && storage->shared() && old.stored) {
                control('D', old.key);
            }
        }
    }
    int tier(const Unit& unit) const {
        if (unit.dead) return 0;
        if (!config.leaf_last) return 1;
        size_t children = 0;
        std::set<int32_t> branches;
        for (const auto& [key, u] : units) {
            (void) key;
            if (u.get() != &unit && !u->dead && u->prefix.tokens.size() > unit.prefix.tokens.size() &&
                extends(u->prefix, unit.prefix)) {
                ++children;
                branches.insert(u->prefix.tokens[unit.prefix.tokens.size()]);
            }
        }
        if ((config.anchor_tier && unit.anchor) || unit.uses > 1 || branches.size() > 1) return 2;
        if (children || (config.idle_seconds > 0 && clock() - unit.used_at >= config.idle_seconds)) return 1;
        return 2;
    }
    std::pair<int, double> score(const Unit& u, bool ram) const {
        const auto size = freeable(u, ram);
        return {tier(u), size ? u.stamp + gain(u) / double(size) : std::numeric_limits<double>::infinity()};
    }
    bool pinned(const std::shared_ptr<Unit>& unit) const { return unit.use_count() > 1 || unit->pending; }

    Bytes manifest(const Unit& u) const {
        Bytes b; put(b, std::string("MAYASTATE3")); put(b, u.prefix.identity);
        put(b, u.prefix.tokens.size()); for (int32_t t : u.prefix.tokens) put(b, uint32_t(t));
        put(b, u.prefix.extra.size());
        for (const auto& x : u.prefix.extra) { put(b, x.position); put(b, x.length); put(b, x.identity); }
        put(b, u.sampling_ready); put(b, u.anchor); put(b, u.frontier); put(b, u.min_prompt_tokens); put(b, u.parts.size());
        for (const auto& p : u.parts) {
            put(b, p.name); put(b, p.chunks.size());
            for (const auto& c : p.chunks) { put(b, c->key); put(b, c->checksum); put(b, c->size); }
        }
        put(b, hash(b)); return b;
    }
    void recover(const std::vector<std::string>* candidates = nullptr) {
        std::string error;
        const auto keys = candidates ? *candidates : storage->manifests(error);
        for (const auto& key : keys) {
            if (units.count(key)) continue;
            if (storage->age_seconds(key) > config.max_age_seconds) continue;
            Bytes b;
            if (!storage->get(key, b, error)) continue;
            try {
                if (b.size() < 8 || key != "unit-" + content_digest(b.data(), b.size()) + ".state") continue;
                Reader r{b};
                if (r.s() != "MAYASTATE3") continue;
                auto u = std::make_shared<Unit>(); u->key = key; u->prefix.identity = r.s();
                const size_t n = r.count(8);
                for (size_t i = 0; i < n; ++i) u->prefix.tokens.push_back(int32_t(r.n()));
                const size_t nx = r.count(24);
                for (size_t i = 0; i < nx; ++i) { ExtraInput x; x.position = r.n(); x.length = r.n(); x.identity = r.s(); u->prefix.extra.push_back(std::move(x)); }
                u->sampling_ready = r.n() != 0; u->anchor = r.n() != 0; u->frontier = r.n() != 0;
                u->min_prompt_tokens = r.n();
                const size_t np = r.count(16);
                for (size_t i = 0; i < np; ++i) {
                    Part p; p.name = r.s(); const size_t nc = r.count(24);
                    for (size_t j = 0; j < nc; ++j) {
                        const std::string ck = r.s(); const auto sum = r.n(), size = r.n();
                        if (!size || size > config.chunk_bytes) throw std::runtime_error("invalid state chunk size");
                        std::shared_ptr<Chunk> found;
                        for (auto& w : chunks[sum]) if (auto c = w.lock(); c && c->key == ck && c->size == size) found = std::move(c);
                        if (!found) { found = std::make_shared<Chunk>(); found->key = ck; found->checksum = sum; found->size = size; found->stored = true; chunks[sum].push_back(found); }
                        p.chunks.push_back(std::move(found));
                    }
                    u->parts.push_back(std::move(p));
                }
                const auto expected = hash(std::span<const uint8_t>(b).first(r.at));
                if (r.n() != expected || r.at != b.size()) continue;
                u->ram = false; u->stored = true; u->imported = candidates != nullptr;
                u->used_at = clock(); units[key] = std::move(u);
            } catch (const std::exception&) { ++stats.read_failures; }
        }
        account();
    }

    void run() {
        for (;;) {
            Job job;
            std::shared_ptr<Unit> u;
            { std::unique_lock lock(mutex); cv.wait(lock, [&] { return stopping || !writes.empty(); });
              if (writes.empty()) { if (stopping) return; else continue; }
              job = std::move(writes.front()); writes.pop_front(); ++active_jobs;
              active_updates += job.operation != 'W'; }
            if (job.operation != 'W') {
                bool ok = true;
                try {
                    if (job.operation == 'T') storage->touch(job.key);
                    else if (job.operation == 'X') storage->invalidate(job.key);
                    else storage->dead(job.key);
                } catch (const std::exception&) { ok = false; }
                { std::lock_guard lock(mutex);
                  --active_jobs; --active_updates;
                  if (!ok) {
                      ++stats.write_failures;
                      const auto found = units.find(job.key);
                      if (job.operation == 'T' && found != units.end() && !found->second->dead && !found->second->imported) {
                          found->second->stored = false;
                          if (!found->second->pending && config.write_through) {
                              found->second->pending = true; writes.emplace_back(found->second);
                          }
                      }
                  }
                  cv.notify_all(); }
                continue;
            }
            u = std::move(job.unit);
            std::string error;
            bool ok = true;
            try {
              if (!job.payload_ready) for (const auto& p : u->parts) for (const auto& c : p.chunks) {
                std::shared_ptr<const Bytes> data;
                { std::lock_guard lock(mutex); if (c->stored && !storage->shared()) continue; data = c->data; }
                if (!data && storage->shared()) {
                    Bytes loaded;
                    if (storage->get(c->key, loaded, error) && loaded.size() == c->size &&
                        content_digest(loaded.data(), loaded.size()) == c->key.substr(6))
                        data = std::make_shared<const Bytes>(std::move(loaded));
                }
                if (!data || !storage->put(c->key, *data, error)) { ok = false; break; }
                { std::lock_guard lock(mutex); c->stored = true; }
              }
              if (ok && !job.payload_ready) { const Bytes b = manifest(*u); ok = storage->put(u->key, b, error); }
              if (ok) {
                  Storage::Publication publication; publication.key = u->key; publication.prefix = u->prefix; publication.frontier = u->frontier;
                  { std::lock_guard lock(mutex);
                    publication.value = config.cost.seconds(0, u->prefix.tokens.size());
                    size_t parent_n = 0;
                    for (const auto& [key, other] : units)
                        if (!other->dead && other->stored && other->prefix.tokens.size() > parent_n &&
                            other->prefix.tokens.size() < u->prefix.tokens.size() && extends(u->prefix, other->prefix)) {
                            publication.parent = key; parent_n = other->prefix.tokens.size();
                        }
                  }
                  for (const auto& part : u->parts) for (const auto& chunk : part.chunks)
                      publication.refs.push_back(chunk->key);
                  std::sort(publication.refs.begin(), publication.refs.end());
                  publication.refs.erase(std::unique(publication.refs.begin(), publication.refs.end()), publication.refs.end());
                  ok = storage->announce(publication, error);
                  bool dead;
                  { std::lock_guard lock(mutex); dead = u->dead; }
                  if (ok && dead && storage->shared()) storage->dead(u->key);
              }
            } catch (const std::exception&) { ok = false; }
            { std::lock_guard lock(mutex); u->pending = false; u->stored = ok; --active_jobs;
              if (!ok) ++stats.write_failures;
              u.reset();
              try { trim(); } catch (const std::exception&) { ++stats.write_failures; }
              cv.notify_all(); }
        }
    }
    void drop(const std::string& key) {
        auto it = units.find(key);
        if (it == units.end()) return;
        const auto owned = holders(*it->second);
        std::vector<std::shared_ptr<Chunk>> retained;
        for (const auto& part : it->second->parts) for (const auto& chunk : part.chunks) retained.push_back(chunk);
        std::vector<std::string> erase;
        for (const auto* c : owned) {
            bool shared = false;
            for (const auto& [k, u] : units) if (k != key && holders(*u).count(c)) { shared = true; break; }
            if (!shared && c->stored) erase.push_back(c->key);
        }
        units.erase(it); ++stats.evicted;
        for (const auto& chunk : retained) {
            bool in_ram = false;
            for (const auto& [other_key, unit] : units) {
                (void) other_key;
                if ((unit->ram || pinned(unit)) && holders(*unit).count(chunk.get())) { in_ram = true; break; }
            }
            if (!in_ram) chunk->data.reset();
        }
        if (storage) {
            try {
                std::string error;
                if (!storage->erase(key, error)) ++stats.write_failures;
                for (const auto& k : erase) if (!storage->erase(k, error)) ++stats.write_failures;
            } catch (const std::exception&) { ++stats.write_failures; }
        }
    }
    void trim() {
        // Dead units are reclaimed even when byte limits have not been reached.
        // Pending writes and reader leases keep both metadata and payload alive.
        for (auto it = units.begin(); it != units.end();) {
            if (it->second->dead && !pinned(it->second)) {
                const auto key = it++->first; drop(key);
            } else ++it;
        }
        for (auto it = chunks.begin(); it != chunks.end();) {
            auto& bucket = it->second;
            bucket.erase(std::remove_if(bucket.begin(), bucket.end(), [](const auto& w) { return w.expired(); }), bucket.end());
            if (bucket.empty()) it = chunks.erase(it); else ++it;
        }
        account();
        for (;;) {
            const bool ram = stats.ram_bytes > config.ram_bytes;
            const bool disk = stats.disk_bytes > config.disk_bytes;
            const bool count = units.size() > config.max_units;
            if (!ram && !disk && !count) break;
            auto best = units.end(); std::pair<int, double> rank{3, std::numeric_limits<double>::infinity()};
            for (auto it = units.begin(); it != units.end(); ++it) {
                if (pinned(it->second) || (ram && !it->second->ram)) continue;
                const auto candidate = score(*it->second, ram);
                if (candidate < rank) { rank = candidate; best = it; }
            }
            if (best == units.end()) break;
            if ((!config.leaf_last || rank.first == 2) && std::isfinite(rank.second))
                stats.watermark = std::max(stats.watermark, rank.second);
            if (ram && best->second->stored && !disk && !count) {
                auto exclusive = holders(*best->second);
                best->second->ram = false;
                for (const auto& [key, u] : units) {
                    (void) key;
                    if (u->ram || pinned(u)) for (const auto* c : holders(*u)) exclusive.erase(c);
                }
                for (const auto* c : exclusive) const_cast<Chunk*>(c)->data.reset();
                ++stats.demoted;
            } else { const std::string key = best->first; drop(key); }
            account();
        }
    }
};

StateCache::StateCache(Config c, std::shared_ptr<Storage> s, std::function<double()> clock)
    : p_(std::make_unique<Impl>(std::move(c), std::move(s), std::move(clock))) {}
StateCache::~StateCache() = default;
uint64_t StateCache::Lease::tokens() const { return unit_ ? unit_->prefix.tokens.size() : 0; }
bool StateCache::Lease::sampling_ready() const { return unit_ && unit_->sampling_ready; }
const Prefix& StateCache::Lease::prefix() const { if (!unit_) throw std::logic_error("empty state lease"); return unit_->prefix; }

bool StateCache::publish(Capture capture, std::string& error) {
    error.clear();
    if (capture.prefix.tokens.empty() || capture.regions.empty()) { error = "state capture is incomplete"; return false; }
    std::lock_guard lock(p_->mutex);
    if (!p_->config.max_units) { ++p_->stats.rejected; return false; }
    uint64_t capture_bytes = 0;
    std::set<std::string> names;
    for (const auto& region : capture.regions) {
        if (region.name.empty() || !names.insert(region.name).second) { error = "duplicate or empty state region name"; return false; }
        if (region.bytes.size() > p_->config.ram_bytes - capture_bytes) { ++p_->stats.rejected; return false; }
        capture_bytes += region.bytes.size();
    }
    if (!capture_bytes) { error = "state capture has no payload"; return false; }
    auto u = std::make_shared<Unit>(); u->prefix = std::move(capture.prefix);
    u->anchor = capture.anchor; u->frontier = capture.frontier; u->sampling_ready = capture.sampling_ready;
    u->min_prompt_tokens = capture.min_prompt_tokens;
    u->stamp = p_->stats.watermark; u->used_at = p_->clock();
    uint64_t total = 0;
    for (auto& r : capture.regions) {
        Part part; part.name = std::move(r.name);
        for (size_t at = 0; at < r.bytes.size(); at += p_->config.chunk_bytes) {
            const auto bytes = std::span<const uint8_t>(r.bytes).subspan(at, std::min(p_->config.chunk_bytes, r.bytes.size() - at));
            const auto sum = hash(bytes);
            const std::string content_key = "chunk-" + content_digest(bytes.data(), bytes.size());
            std::shared_ptr<Chunk> c;
            auto& bucket = p_->chunks[sum];
            for (auto it = bucket.begin(); it != bucket.end();) {
                auto existing = it->lock();
                if (!existing) { it = bucket.erase(it); continue; }
                if (existing->key == content_key && existing->size == bytes.size()) {
                    c = std::move(existing);
                    if (!c->data) c->data = std::make_shared<const Bytes>(bytes.begin(), bytes.end());
                    break;
                }
                ++it;
            }
            if (!c) {
                c = std::make_shared<Chunk>(); c->checksum = sum; c->size = bytes.size();
                c->key = content_key;
                c->data = std::make_shared<const Bytes>(bytes.begin(), bytes.end()); bucket.push_back(c);
            }
            total += c->size; part.chunks.push_back(std::move(c));
        }
        u->parts.push_back(std::move(part));
    }
    if (total > p_->config.ram_bytes || p_->config.max_units == 0) { ++p_->stats.rejected; return false; }
    const Bytes m = p_->manifest(*u);
    for (auto& [key, existing] : p_->units) {
        (void) key;
        if (!existing->dead && existing->prefix.identity == u->prefix.identity && existing->prefix.tokens == u->prefix.tokens &&
            existing->prefix.extra == u->prefix.extra && p_->manifest(*existing) == m) {
            existing->stamp = p_->stats.watermark; existing->used_at = p_->clock(); existing->ram = true;
            if (p_->storage && !existing->stored && !existing->pending && p_->config.write_through &&
                (p_->config.disk_bytes || p_->storage->shared())) {
                existing->pending = true; p_->writes.emplace_back(existing); p_->cv.notify_one();
            }
            const std::string existing_key = key;
            p_->trim();
            const bool retained = p_->units.count(existing_key) != 0;
            if (!retained) ++p_->stats.rejected;
            return retained;
        }
    }
    u->key = "unit-" + content_digest(m.data(), m.size()) + ".state";
    const auto key = u->key;
    p_->units.emplace(key, std::move(u));
    p_->trim();
    const auto found = p_->units.find(key);
    if (found == p_->units.end() || p_->stats.ram_bytes > p_->config.ram_bytes) {
        if (found != p_->units.end()) p_->drop(key);
        ++p_->stats.rejected; p_->account(); return false;
    }
    ++p_->stats.admitted;
    if (p_->storage && p_->config.write_through && (p_->config.disk_bytes || p_->storage->shared())) {
        found->second->pending = true; p_->writes.push_back(found->second); p_->cv.notify_one();
    }
    p_->retire_frontiers(*found->second);
    p_->trim();
    return true;
}

bool StateCache::publish(Capture metadata, const std::vector<RegionSource>& sources, std::string& error) {
    error.clear();
    if (metadata.prefix.tokens.empty() || sources.empty() || !metadata.regions.empty()) {
        error = "streaming capture metadata is incomplete"; return false;
    }
    std::unique_lock lock(p_->mutex);
    ++p_->stats.publishes;
    Elapsed elapsed{p_->stats.publish_seconds};
    if (!p_->config.max_units) { ++p_->stats.rejected; return false; }
    struct Read {
        std::shared_ptr<Chunk> chunk;
        size_t source;
        uint64_t offset;
        std::shared_ptr<const Bytes> captured;
    };
    std::vector<Read> reads;
    std::vector<std::shared_ptr<Chunk>> reserved;
    std::string key;
    bool complete = false;
    // The cache lock keeps policy, shared backing and source admission coherent
    // throughout this synchronous native capture. Storage never calls the cache.
    struct Cleanup {
        Impl& cache;
        std::vector<std::shared_ptr<Chunk>>& chunks;
        std::string& key;
        bool& complete;
        ~Cleanup() {
            if (!complete && !key.empty()) {
                try { cache.drop(key); }
                catch (...) { cache.units.erase(key); ++cache.stats.write_failures; }
            }
            for (const auto& c : chunks) c->reserved_ram = c->reserved_disk = false;
            try { cache.account(); } catch (...) { ++cache.stats.write_failures; }
        }
    } cleanup{*p_, reserved, key, complete};
    try {
        auto unit = std::make_shared<Unit>();
        unit->prefix = std::move(metadata.prefix); unit->anchor = metadata.anchor;
        unit->frontier = metadata.frontier; unit->sampling_ready = metadata.sampling_ready;
        unit->min_prompt_tokens = metadata.min_prompt_tokens;
        unit->stamp = p_->stats.watermark; unit->used_at = p_->clock();
        std::set<std::string> names;
        std::set<Chunk*> seen;
        uint64_t unique = 0;
        bool capture_payload = p_->stats.ram_bytes <= p_->config.ram_bytes;
        uint64_t headroom = capture_payload ? p_->config.ram_bytes - p_->stats.ram_bytes : 0;
        for (const auto& source : sources) {
            if (source.bytes > headroom) { capture_payload = false; break; }
            headroom -= source.bytes;
        }
        if (capture_payload) for (const auto& [key, existing] : p_->units) {
            (void) key;
            if (existing->prefix.identity == unit->prefix.identity && existing->prefix.tokens == unit->prefix.tokens &&
                existing->prefix.extra == unit->prefix.extra) {
                capture_payload = false; break;
            }
        }
        size_t hash_workers = 0;
        if (capture_payload && p_->config.ram_bytes - p_->stats.ram_bytes - headroom >= (4ull << 20)) {
            size_t chunks = 0;
            for (const auto& source : sources) {
                if (source.bytes) chunks += source.bytes > p_->config.chunk_bytes ? 2 : 1;
                if (chunks >= 2) break;
            }
            if (chunks >= 2) hash_workers = std::min(16u, std::thread::hardware_concurrency());
        }
        const bool parallel_hash = hash_workers >= 2;
        struct PreparedChunk {
            std::shared_ptr<const Bytes> bytes;
            uint64_t checksum;
            std::string key;
        };
        std::vector<PreparedChunk> prepared;
        if (parallel_hash) {
            for (const auto& source : sources) {
                if (source.name.empty() || !names.insert(source.name).second || !source.read) {
                    error = "duplicate or invalid streaming state region"; return false;
                }
                for (uint64_t at = 0; at < source.bytes;) {
                    const size_t count = size_t(std::min<uint64_t>(p_->config.chunk_bytes, source.bytes-at));
                    auto bytes = std::make_shared<Bytes>(count);
                    if (!source.read(at, bytes->data(), count, error)) return false;
                    p_->stats.read_bytes += count;
                    prepared.push_back({std::move(bytes), 0, {}}); at += count;
                }
            }
            hash_workers = std::min(hash_workers, prepared.size());
            std::vector<std::future<void>> jobs;
            jobs.reserve(hash_workers);
            for (size_t worker = 0; worker < hash_workers; ++worker) {
                jobs.push_back(std::async(std::launch::async, [&prepared, worker, hash_workers] {
                    for (size_t at = worker; at < prepared.size(); at += hash_workers) {
                        auto& chunk = prepared[at];
                        chunk.checksum = hash(*chunk.bytes);
                        chunk.key = "chunk-" + content_digest(chunk.bytes->data(), chunk.bytes->size());
                    }
                }));
            }
            for (auto& job : jobs) job.get();
        }
        Bytes scratch(parallel_hash ? 0 : p_->config.chunk_bytes);
        size_t prepared_at = 0;
        for (size_t i = 0; i < sources.size(); ++i) {
            const auto& source = sources[i];
            if (!parallel_hash && (source.name.empty() || !names.insert(source.name).second || !source.read)) {
                error = "duplicate or invalid streaming state region"; return false;
            }
            Part part; part.name = source.name;
            for (uint64_t at = 0; at < source.bytes;) {
                const size_t count = size_t(std::min<uint64_t>(p_->config.chunk_bytes, source.bytes-at));
                const auto* staged = parallel_hash ? &prepared[prepared_at++] : nullptr;
                if (!staged && !source.read(at, scratch.data(), count, error)) return false;
                if (!staged) p_->stats.read_bytes += count;
                p_->stats.hashed_bytes += count;
                const auto data = staged ? std::span<const uint8_t>(*staged->bytes) :
                    std::span<const uint8_t>(scratch.data(), count);
                const auto checksum = staged ? staged->checksum : hash(data);
                const auto chunk_key = staged ? staged->key : "chunk-" + content_digest(data.data(), data.size());
                std::shared_ptr<Chunk> chunk;
                auto& bucket = p_->chunks[checksum];
                for (auto it = bucket.begin(); it != bucket.end();) {
                    auto old = it->lock();
                    if (!old) { it = bucket.erase(it); continue; }
                    if (old->key == chunk_key && old->size == count) { chunk = std::move(old); break; }
                    ++it;
                }
                if (!chunk) {
                    chunk = std::make_shared<Chunk>(); chunk->key = chunk_key;
                    chunk->checksum = checksum; chunk->size = count; bucket.push_back(chunk);
                }
                if (seen.insert(chunk.get()).second) {
                    if (count > UINT64_MAX-unique) { error = "state byte count overflow"; return false; }
                    unique += count;
                    std::shared_ptr<const Bytes> captured;
                    if (capture_payload && !chunk->data) captured = staged ? staged->bytes :
                        std::make_shared<const Bytes>(data.begin(), data.end());
                    reads.push_back({chunk, i, at, std::move(captured)});
                }
                part.chunks.push_back(std::move(chunk)); at += count;
            }
            unit->parts.push_back(std::move(part));
        }
        if (!unique) { error = "state capture has no payload"; return false; }
        unit->ram = unique <= p_->config.ram_bytes;
        if (!unit->ram && (!p_->storage || (!p_->config.disk_bytes && !p_->storage->shared()))) {
            ++p_->stats.rejected; return false;
        }
        const auto manifest = p_->manifest(*unit);
        const auto candidate = "unit-" + content_digest(manifest.data(), manifest.size()) + ".state";
        if (const auto existing = p_->units.find(candidate); existing != p_->units.end()) {
            if (existing->second->dead) { ++p_->stats.rejected; return false; }
            existing->second->stamp = p_->stats.watermark; existing->second->used_at = p_->clock();
            if (p_->storage && existing->second->stored) p_->control('T', candidate);
            complete = true; return true;
        }
        for (const auto& read : reads) {
            const auto& chunk = read.chunk;
            chunk->reserved_ram = unit->ram && !chunk->data;
            chunk->reserved_disk = !unit->ram && p_->storage->has_local_tier() &&
                p_->storage->local_bytes(chunk->key, chunk->size) != chunk->size;
            reserved.push_back(chunk);
        }
        unit->key = candidate; key = candidate;
        p_->units.emplace(key, std::move(unit));
        p_->trim();
        const auto retained = p_->units.find(key);
        if (retained == p_->units.end()) { ++p_->stats.rejected; key.clear(); return false; }
        Unit& target = *retained->second;
        target.pending = true;  // pin all members while their reserved bytes become real
        for (const auto& read : reads) {
            auto& chunk = *read.chunk;
            const bool store = !target.ram && (!chunk.stored || p_->storage->shared() ||
                (p_->storage->has_local_tier() && p_->storage->local_bytes(chunk.key, chunk.size) != chunk.size));
            if ((!target.ram || chunk.data) && !store) continue;
            std::shared_ptr<const Bytes> data = chunk.data;
            if (!data) data = read.captured;
            if (!data) {
                auto bytes = std::make_shared<Bytes>(size_t(chunk.size));
                if (!sources[read.source].read(read.offset, bytes->data(), bytes->size(), error)) {
                    ++p_->stats.rejected; target.pending = false; return false;
                }
                p_->stats.read_bytes += bytes->size(); p_->stats.hashed_bytes += bytes->size();
                if (hash(*bytes) != chunk.checksum || "chunk-" + content_digest(bytes->data(), bytes->size()) != chunk.key) {
                    error = "native state changed during capture"; ++p_->stats.rejected; target.pending = false; return false;
                }
                data = std::move(bytes);
            }
            if (target.ram) { chunk.data = std::move(data); chunk.reserved_ram = false; }
            else {
                if (!p_->storage->put(chunk.key, *data, error)) {
                    ++p_->stats.write_failures; target.pending = false; return false;
                }
                chunk.stored = true; chunk.reserved_disk = false;
            }
        }
        target.pending = false;
        if (!target.ram) {
            if (!p_->storage->put(key, manifest, error)) { ++p_->stats.write_failures; return false; }
            target.stored = true;
        }
        ++p_->stats.admitted;
        if (p_->storage && (!target.ram || p_->config.write_through) &&
            (p_->config.disk_bytes || p_->storage->shared())) {
            target.pending = true;
            Impl::Job job(retained->second); job.payload_ready = target.stored;
            p_->writes.push_back(std::move(job)); p_->cv.notify_one();
        }
        p_->retire_frontiers(target);
        complete = true; return true;
    } catch (const std::exception& e) {
        if (!key.empty()) {
            const auto found = p_->units.find(key);
            if (found != p_->units.end()) found->second->pending = false;
        }
        error = std::string("state capture failed: ") + e.what(); ++p_->stats.rejected; return false;
    }
}

StateCache::Lease StateCache::find_resume(const Prefix& prompt, bool need_sample) {
    std::lock_guard lock(p_->mutex);
    if (p_->storage && p_->storage->shared()) {
        try {
            std::string error;
            const auto candidates = p_->storage->lookup(prompt, error);
            if (!error.empty()) ++p_->stats.read_failures;
            p_->recover(&candidates);
        } catch (const std::exception&) { ++p_->stats.read_failures; }
    }
    Lease found;
    for (const auto& [key, u] : p_->units) {
        (void) key;
        if (u->dead || u->prefix.tokens.size() < found.tokens() || prompt.tokens.size() < u->min_prompt_tokens || !extends(prompt, u->prefix) ||
            (need_sample && u->prefix.tokens.size() == prompt.tokens.size() && !u->sampling_ready)) continue;
        if (u->imported) {
            std::string error;
            if (!p_->storage->validate_unit(key, error)) {
                u->dead = true; ++p_->stats.read_failures; continue;
            }
        }
        found.unit_ = u;
    }
    if (found) { found.unit_->stamp = p_->stats.watermark; found.unit_->used_at = p_->clock(); ++found.unit_->uses;
                 ++p_->stats.hits;
                 bool ram = true, disk = bool(p_->storage);
                 for (const auto& part : found.unit_->parts) for (const auto& chunk : part.chunks) {
                     ram = ram && bool(chunk->data);
                     disk = disk && p_->storage->local_bytes(chunk->key, chunk->size) == chunk->size;
                 }
                 if (ram) ++p_->stats.ram_hits;
                 else if (disk) ++p_->stats.disk_hits;
                 else ++p_->stats.remote_hits;
                 if (p_->storage && found.unit_->stored) {
                     p_->control('T', found.unit_->key);
                 }
    }
    else ++p_->stats.misses;
    p_->trim();
    return found;
}

bool StateCache::restore(const Lease& lease, std::vector<Region>& regions, std::string& error) {
    error.clear(); regions.clear();
    if (!lease) { error = "empty state lease"; return false; }
    { std::lock_guard lock(p_->mutex);
      if (lease.unit_->dead) { error = "state recovery unit was invalidated"; return false; } }
    uint64_t total = 0;
    for (const auto& part : lease.unit_->parts) for (const auto& chunk : part.chunks) {
        if (chunk->size > p_->config.ram_bytes - total) { error = "state restore exceeds host budget"; return false; }
        total += chunk->size;
    }
    try {
    std::vector<Region> loaded;
    std::map<Chunk*, std::shared_ptr<const Bytes>> fetched;
    for (const auto& part : lease.unit_->parts) {
        Region region; region.name = part.name;
        for (const auto& c : part.chunks) {
            std::shared_ptr<const Bytes> bytes;
            { std::lock_guard lock(p_->mutex); bytes = c->data; }
            Bytes disk;
            if (!bytes) {
                const auto previous = fetched.find(c.get());
                if (previous != fetched.end()) bytes = previous->second;
                else if (!p_->storage || !p_->storage->get(c->key, disk, error) || disk.size() != c->size || hash(disk) != c->checksum ||
                         c->key != "chunk-" + content_digest(disk.data(), disk.size())) {
                    std::lock_guard lock(p_->mutex); ++p_->stats.read_failures; lease.unit_->dead = true;
                    if (error.empty()) error = "state object checksum or size mismatch";
                    return false;
                }
                if (!bytes) { bytes = std::make_shared<const Bytes>(std::move(disk)); fetched[c.get()] = bytes; }
            }
            region.bytes.insert(region.bytes.end(), bytes->begin(), bytes->end());
        }
        loaded.push_back(std::move(region));
    }
    {
        std::lock_guard lock(p_->mutex);
        if (lease.unit_->dead) { error = "state recovery unit was invalidated during load"; return false; }
        if (!fetched.empty()) {
            std::vector<Chunk*> promoted;
            for (const auto& [chunk, bytes] : fetched) if (!chunk->data) { chunk->data = bytes; promoted.push_back(chunk); }
            const bool old_ram = lease.unit_->ram;
            lease.unit_->ram = true;
            p_->trim();
            if (p_->stats.ram_bytes > p_->config.ram_bytes) {
                for (auto* chunk : promoted) chunk->data.reset();
                lease.unit_->ram = old_ram; p_->account();
            } else ++p_->stats.promoted;
        }
    }
    regions = std::move(loaded);
    return true;
    } catch (const std::exception& e) {
        error = std::string("state restore failed: ") + e.what();
        std::lock_guard lock(p_->mutex); ++p_->stats.read_failures;
        return false;
    }
}
bool StateCache::restore(const Lease& lease, const std::vector<RegionTarget>& targets, std::string& error) {
    error.clear();
    if (!lease) { error = "empty state lease"; return false; }
    struct Timed {
        Impl& cache;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        ~Timed() {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::lock_guard lock(cache.mutex); ++cache.stats.restores; cache.stats.restore_seconds += s;
        }
    } timed{*p_};
    { std::lock_guard lock(p_->mutex);
      if (lease.unit_->dead) { error = "state recovery unit was invalidated"; return false; } }
    try {
        std::map<std::string, const Part*> parts;
        for (const auto& part : lease.unit_->parts)
            if (!parts.emplace(part.name, &part).second) { error = "duplicate recovery region"; return false; }
        std::set<std::string> names;
        if (parts.size() != targets.size()) { error = "incomplete native recovery layout"; return false; }
        for (const auto& target : targets) {
            const auto found = parts.find(target.name);
            uint64_t size = 0;
            if (found == parts.end() || !target.write || !names.insert(target.name).second) {
                error = "invalid native recovery region"; return false;
            }
            for (const auto& chunk : found->second->chunks) {
                if (chunk->size > UINT64_MAX-size) { error = "recovery byte count overflow"; return false; }
                size += chunk->size;
            }
            if (size != target.bytes) { error = "native recovery size mismatch: " + target.name; return false; }
        }
        uint64_t unique_bytes = 0;
        std::set<const Chunk*> unique;
        bool cache_payload = true, promoted = false;
        for (const auto& part : lease.unit_->parts) for (const auto& chunk : part.chunks)
            if (unique.insert(chunk.get()).second) {
                if (chunk->size > p_->config.ram_bytes-unique_bytes) cache_payload = false;
                else unique_bytes += chunk->size;
            }
        for (const auto& target : targets) {
            uint64_t at = 0;
            const auto& chunks = parts.at(target.name)->chunks;
            for (size_t begin = 0; begin < chunks.size();) {
                size_t end = begin;
                uint64_t batch_bytes = 0;
                do { batch_bytes += chunks[end++]->size; }
                while (end < chunks.size() && chunks[end]->size <= (4ull << 20) - std::min<uint64_t>(batch_bytes, 4ull << 20));
                std::vector<std::shared_ptr<const Bytes>> data(end-begin);
                std::vector<std::string> missing;
                std::vector<size_t> indices;
                {
                    std::lock_guard lock(p_->mutex);
                    if (lease.unit_->dead) { error = "state invalidated during native load"; return false; }
                    for (size_t i = begin; i < end; ++i) {
                        data[i-begin] = chunks[i]->data;
                        if (!data[i-begin]) { missing.push_back(chunks[i]->key); indices.push_back(i-begin); }
                    }
                }
                if (!missing.empty()) {
                    std::vector<Bytes> fetched;
                    bool ok = p_->storage && p_->storage->get_chunks(lease.unit_->key, missing, fetched, error) && fetched.size() == missing.size();
                    for (size_t i = 0; ok && i < fetched.size(); ++i) {
                        const auto& chunk = chunks[begin+indices[i]];
                        ok = fetched[i].size() == chunk->size && hash(fetched[i]) == chunk->checksum &&
                             "chunk-" + content_digest(fetched[i].data(), fetched[i].size()) == chunk->key;
                    }
                    if (!ok) {
                        std::lock_guard lock(p_->mutex); ++p_->stats.read_failures; lease.unit_->dead = true;
                        if (error.empty()) error = "incomplete or corrupt native recovery unit";
                        return false;
                    }
                    for (size_t i = 0; i < fetched.size(); ++i) data[indices[i]] = std::make_shared<const Bytes>(std::move(fetched[i]));
                }
                for (const auto& bytes : data) {
                    if (!target.write(at, bytes->data(), bytes->size(), error)) return false;
                    at += bytes->size();
                }
                // Small recovery units can return to RAM without retaining an
                // extra full-unit buffer. Account each verified batch before the
                // next read, and roll it back if active pins prevent admission.
                if (cache_payload) {
                    std::lock_guard lock(p_->mutex);
                    if (lease.unit_->dead) { error = "state invalidated during native load"; return false; }
                    std::vector<Chunk*> fresh;
                    for (size_t i = begin; i < end; ++i) if (!chunks[i]->data) {
                        chunks[i]->data = data[i-begin]; fresh.push_back(chunks[i].get());
                    }
                    if (!fresh.empty()) {
                        const bool before = lease.unit_->ram;
                        lease.unit_->ram = true; p_->trim();
                        if (p_->stats.ram_bytes > p_->config.ram_bytes) {
                            for (auto* chunk : fresh) chunk->data.reset();
                            lease.unit_->ram = before; p_->account();
                        } else promoted = true;
                    }
                }
                begin = end;
            }
        }
        if (promoted) { std::lock_guard lock(p_->mutex); ++p_->stats.promoted; }
        return true;
    } catch (const std::exception& e) {
        error = std::string("native state load failed: ") + e.what();
        std::lock_guard lock(p_->mutex); ++p_->stats.read_failures; return false;
    }
}

bool StateCache::read_region(const Lease& lease, const std::string& name, Bytes& bytes, std::string& error, uint64_t limit) {
    bytes.clear(); error.clear();
    if (!lease) { error = "empty state lease"; return false; }
    { std::lock_guard lock(p_->mutex);
      if (lease.unit_->dead) { error = "invalidated state lease"; return false; } }
    const Part* found = nullptr;
    for (const auto& part : lease.unit_->parts) if (part.name == name) {
        if (found) { error = "duplicate recovery region"; return false; }
        found = &part;
    }
    if (!found) { error = "recovery sideband missing: " + name; return false; }
    uint64_t size = 0;
    for (const auto& chunk : found->chunks) {
        if (chunk->size > limit-size) { error = "recovery sideband exceeds bound"; return false; }
        size += chunk->size;
    }
    try {
        Bytes result; result.reserve(size_t(size));
        for (const auto& chunk : found->chunks) {
            std::shared_ptr<const Bytes> data;
            { std::lock_guard lock(p_->mutex); data = chunk->data; }
            if (!data) {
                auto loaded = std::make_shared<Bytes>();
                if (!p_->storage || !p_->storage->get_chunk(lease.unit_->key, chunk->key, *loaded, error) ||
                    loaded->size() != chunk->size || hash(*loaded) != chunk->checksum ||
                    "chunk-" + content_digest(loaded->data(), loaded->size()) != chunk->key) {
                    if (error.empty()) error = "corrupt recovery sideband";
                    return false;
                }
                data = std::move(loaded);
            }
            result.insert(result.end(), data->begin(), data->end());
        }
        bytes = std::move(result); return true;
    } catch (const std::exception& e) { error = std::string("recovery sideband failed: ") + e.what(); return false; }
}

void StateCache::restored(Lease& lease) {
    std::lock_guard lock(p_->mutex);
    if (lease && !lease.restored_ && !lease.unit_->dead) {
        p_->stats.restored_tokens += lease.tokens(); lease.restored_ = true;
    }
}
void StateCache::release(Lease& lease) {
    std::lock_guard lock(p_->mutex);
    if (lease.unit_) { lease.unit_->stamp = p_->stats.watermark; lease.unit_->used_at = p_->clock(); lease.unit_.reset(); }
    p_->trim();
}
void StateCache::supersede(const Prefix& newest, size_t keep) {
    std::lock_guard lock(p_->mutex);
    std::vector<Unit*> ancestors;
    for (auto& [key, u] : p_->units) {
        (void) key;
        if (!u->anchor && u->prefix.tokens.size() < newest.tokens.size() && extends(newest, u->prefix)) ancestors.push_back(u.get());
    }
    std::sort(ancestors.begin(), ancestors.end(), [](const Unit* a, const Unit* b) { return a->prefix.tokens.size() > b->prefix.tokens.size(); });
    for (size_t i = keep; i < ancestors.size(); ++i) if (p_->tier(*ancestors[i]) < 2) {
        ancestors[i]->dead = true;
        if (p_->storage && p_->storage->shared()) {
            p_->control('D', ancestors[i]->key);
        }
    }
    p_->trim();
}
void StateCache::flush() {
    std::unique_lock lock(p_->mutex);
    p_->cv.wait(lock, [&] { return p_->writes.empty() && !p_->active_jobs; });
}
void StateCache::invalidate(const Prefix& failed) {
    std::lock_guard lock(p_->mutex);
    for (auto& [key, u] : p_->units) {
        (void) key;
        if (extends(failed, u->prefix) || extends(u->prefix, failed)) {
            u->dead = true;
            if (p_->storage && p_->storage->shared()) {
                p_->control('X', key);
            }
        }
    }
    p_->trim();
}
Stats StateCache::stats() const { std::lock_guard lock(p_->mutex); p_->account(); return p_->stats; }
std::string StateCache::report() const {
    try {
    const auto s = stats();
    std::ostringstream out;
    out << std::setprecision(12)
        << "hits=" << s.hits << " misses=" << s.misses << " restored_tokens=" << s.restored_tokens
        << " units=" << s.units << " ram_bytes=" << s.ram_bytes << " disk_bytes=" << s.disk_bytes
        << " logical_bytes=" << s.logical_bytes << " admitted=" << s.admitted << " rejected=" << s.rejected
        << " evicted=" << s.evicted << " promoted=" << s.promoted << " demoted=" << s.demoted
        << " read_failures=" << s.read_failures << " write_failures=" << s.write_failures
        << " watermark=" << s.watermark << " cost_d=" << s.cost_per_token << " cost_e=" << s.cost_per_token_per_100k
        << " calibration_samples=" << s.calibration_samples << " pending_writes=" << s.pending_writes
        << " pending_updates=" << s.pending_updates;
    out << " publishes=" << s.publishes << " publish_ms=" << s.publish_seconds * 1000 << " read_bytes=" << s.read_bytes
        << " hashed_bytes=" << s.hashed_bytes << " restores=" << s.restores << " restore_ms=" << s.restore_seconds * 1000;
    out << " unique_bytes=" << s.unique_bytes << " shared_bytes=" << s.shared_bytes << " pinned_units=" << s.pinned_units
        << " ram_hits=" << s.ram_hits << " disk_hits=" << s.disk_hits << " remote_hits=" << s.remote_hits;
    return out.str();
    } catch (const std::bad_alloc&) { return {}; }
}

void StateCache::observe_prefill(uint64_t from, uint64_t to, double seconds) {
    if (to <= from || !std::isfinite(seconds) || seconds <= 0) return;
    std::lock_guard lock(p_->mutex);
    if (!p_->config.calibrate || p_->stats.calibration_samples >= 16) return;
    const double x = double(to - from), y = x * (double(to) + double(from)) / 200000.;
    p_->cost_xx += x * x; p_->cost_xy += x * y; p_->cost_yy += y * y;
    p_->cost_xt += x * seconds; p_->cost_yt += y * seconds;
    ++p_->stats.calibration_samples;
    double d = p_->cost_xt / p_->cost_xx, e = 0;
    const double determinant = p_->cost_xx * p_->cost_yy - p_->cost_xy * p_->cost_xy;
    if (determinant > 1e-10 * p_->cost_xx * p_->cost_yy) {
        d = (p_->cost_xt * p_->cost_yy - p_->cost_yt * p_->cost_xy) / determinant;
        e = (p_->cost_yt * p_->cost_xx - p_->cost_xt * p_->cost_xy) / determinant;
        if (e < 0) { e = 0; d = p_->cost_xt / p_->cost_xx; }
        if (d < 0) { d = 0; e = p_->cost_yt / p_->cost_yy; }
    }
    if (!std::isfinite(d) || !std::isfinite(e) || d + e <= 0) return;
    p_->config.cost = {d, e};
    p_->stats.cost_per_token = d; p_->stats.cost_per_token_per_100k = e;
    // During initial calibration, clock stamps and gains must use the same units.
    // Freeze the fitted model after sixteen observations; ordinary aging follows.
    p_->stats.watermark = 0;
    for (auto& [key, unit] : p_->units) { (void) key; unit->stamp = 0; }
}

ExportPlan plan_export(uint64_t start, uint64_t end, const std::vector<uint64_t>& marks, uint64_t alignment, bool supported) {
    ExportPlan plan{std::max(start, end), 0};
    std::vector<uint64_t> inner;
    for (auto p : marks) if (p > start && p < end) inner.push_back(p);
    std::sort(inner.begin(), inner.end()); inner.erase(std::unique(inner.begin(), inner.end()), inner.end());
    if (inner.empty()) return plan;
    if (!supported || !alignment || (inner[0] - start) % alignment) return {inner[0], 0};
    return {inner.size() > 1 ? inner[1] : end, inner[0]};
}
bool Commit::exact(uint64_t& endpoint) const {
    if (!completed || kept != accepted || accepted > processed || start > std::numeric_limits<uint64_t>::max() - accepted) return false;
    endpoint = start + accepted; return true;
}

}  // namespace strata::core::statecache
