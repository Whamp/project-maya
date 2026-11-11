#include "strata/core/state_cache.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

extern char** environ;
namespace strata::core::statecache {
namespace {
void number(Bytes& out, uint64_t n) { for (int i = 0; i < 8; ++i) out.push_back(uint8_t(n >> (i*8))); }
void string(Bytes& out, const std::string& s) { number(out, s.size()); out.insert(out.end(), s.begin(), s.end()); }
Bytes encode(const Prefix& p) {
    Bytes b; string(b, p.identity); number(b, p.tokens.size());
    for (auto t : p.tokens) number(b, uint32_t(t));
    number(b, p.extra.size());
    for (const auto& x : p.extra) { number(b, x.position); number(b, x.length); string(b, x.identity); }
    return b;
}
class Mooncake final : public Storage {
    std::shared_ptr<Storage> local_;
    int socket_ = -1;
    pid_t child_ = -1;
    std::mutex mutex_;
    int timeout_ms_ = 10000;
    void stop() {
        if (socket_ >= 0) { close(socket_); socket_ = -1; }
        if (child_ > 0) {
            // Only the helper created by this object is ever terminated.
            kill(child_, SIGTERM);
            if (waitpid(child_, nullptr, WNOHANG) == 0) {
                kill(child_, SIGKILL); while (waitpid(child_, nullptr, 0) < 0 && errno == EINTR) {}
            }
            child_ = -1;
        }
    }
    void transfer(uint8_t* data, size_t n, bool write, std::chrono::steady_clock::time_point end) {
        while (n) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(end-std::chrono::steady_clock::now()).count();
            if (remaining <= 0) throw std::runtime_error("Mooncake helper timeout");
            pollfd p{socket_, short(write ? POLLOUT : POLLIN), 0};
            const int ready = poll(&p, 1, int(std::min<int64_t>(remaining, 60000)));
            if (ready < 0 && errno == EINTR) continue;
            if (ready <= 0) throw std::runtime_error("Mooncake helper unavailable");
            const auto done = write ? send(socket_, data, n, MSG_NOSIGNAL) : recv(socket_, data, n, 0);
            if (done < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            if (done <= 0) throw std::runtime_error("Mooncake helper disconnected");
            data += done; n -= size_t(done);
        }
    }
    Bytes request(char op, const std::string& key, const Bytes& data) {
        std::lock_guard lock(mutex_);
        if (socket_ < 0) throw std::runtime_error("Mooncake helper is offline");
        Bytes body{uint8_t(op)}; string(body, key); body.insert(body.end(), data.begin(), data.end());
        Bytes size; number(size, body.size());
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);
        Bytes result;
        try {
            transfer(size.data(), 8, true, end); transfer(body.data(), body.size(), true, end);
            transfer(size.data(), 8, false, end);
            uint64_t count = 0; for (int i = 0; i < 8; ++i) count |= uint64_t(size[size_t(i)]) << (i*8);
            if (!count || count > (64ull << 20)) throw std::runtime_error("invalid Mooncake response length");
            result.resize(size_t(count)); transfer(result.data(), result.size(), false, end);
        } catch (...) { stop(); throw; }
        if (result.front()) throw std::runtime_error(std::string(result.begin()+1, result.end()));
        result.erase(result.begin()); return result;
    }
public:
    explicit Mooncake(std::shared_ptr<Storage> local) : local_(std::move(local)) {
        int sockets[2];
        int socket_type = SOCK_STREAM;
#ifdef SOCK_CLOEXEC
        socket_type |= SOCK_CLOEXEC;
#endif
        if (socketpair(AF_UNIX, socket_type, 0, sockets)) throw std::runtime_error("Mooncake socketpair failed");
#ifndef SOCK_CLOEXEC
        // Darwin exposes close-on-exec through fcntl rather than socket type flags.
        if (fcntl(sockets[0], F_SETFD, FD_CLOEXEC) < 0 || fcntl(sockets[1], F_SETFD, FD_CLOEXEC) < 0) {
            close(sockets[0]); close(sockets[1]);
            throw std::runtime_error("Mooncake close-on-exec setup failed");
        }
#endif
        const char* python = std::getenv("MAYA_STATE_MOONCAKE_PYTHON");
        const char* helper = std::getenv("MAYA_STATE_MOONCAKE_HELPER");
        if (!python) python = "python3";
        if (!helper) helper = STRATA_STATE_MOONCAKE_HELPER;
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addclose(&actions, sockets[0]);
        posix_spawn_file_actions_adddup2(&actions, sockets[1], 3);
        if (sockets[1] != 3) posix_spawn_file_actions_addclose(&actions, sockets[1]);
        posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO);
        char* argv[] = {const_cast<char*>(python), const_cast<char*>(helper), const_cast<char*>("--fd"), const_cast<char*>("3"), nullptr};
        const int rc = posix_spawnp(&child_, python, &actions, nullptr, argv, environ);
        posix_spawn_file_actions_destroy(&actions); close(sockets[1]); socket_ = sockets[0];
        if (rc) { stop(); throw std::runtime_error("cannot start Mooncake helper"); }
#ifdef SO_NOSIGPIPE
        int no_sigpipe = 1;
        setsockopt(socket_, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof no_sigpipe);
#endif
        fcntl(socket_, F_SETFL, fcntl(socket_, F_GETFL) | O_NONBLOCK);
        if (const char* t = std::getenv("MAYA_STATE_MOONCAKE_TIMEOUT_MS")) timeout_ms_ = std::clamp(std::atoi(t), 100, 60000);
        try {
            auto hello = request('H', "", {});
            if (std::string(hello.begin(), hello.end()) != "MAYA_MOONCAKE1") throw std::runtime_error("Mooncake capability handshake failed");
        } catch (...) { stop(); throw; }
    }
    ~Mooncake() override { stop(); }
    bool shared() const override { return true; }
    bool validate_unit(const std::string& key, std::string& error) override {
        // Imported RAM/SSD copies do not override the producer's revocation.
        try { request('V', key, {}); return true; }
        catch (const std::exception& e) { error = e.what(); return false; }
    }
    bool has_local_tier() const override { return bool(local_); }
    uint64_t local_bytes(const std::string& key, uint64_t expected) const override {
        return local_ ? local_->local_bytes(key, expected) : 0;
    }
    double age_seconds(const std::string& key) const override { return local_ ? local_->age_seconds(key) : 0; }
    bool put(const std::string& key, const Bytes& data, std::string& error) override {
        try {
            if (local_ && !local_->put(key, data, error)) return false;
            request('P', key, data); return true;
        } catch (const std::exception& e) { error = e.what(); return false; }
    }
    bool get(const std::string& key, Bytes& data, std::string& error) override {
        if (local_ && local_->get(key, data, error)) { error.clear(); return true; }
        try { data = request('G', key, {}); error.clear(); return true; }
        catch (const std::exception& e) { error = e.what(); return false; }
    }
    bool get_chunk(const std::string& unit, const std::string& key, Bytes& data, std::string& error) override {
        if (local_ && local_->get(key, data, error)) { error.clear(); return true; }
        try { data = request('G', key, Bytes(unit.begin(), unit.end())); error.clear(); return true; }
        catch (const std::exception& e) { error = e.what(); return false; }
    }
    bool get_chunks(const std::string& unit, const std::vector<std::string>& keys,
                    std::vector<Bytes>& data, std::string& error) override {
        data.clear(); data.resize(keys.size());
        std::vector<size_t> missing;
        for (size_t i = 0; i < keys.size(); ++i)
            if (!local_ || !local_->get(keys[i], data[i], error)) missing.push_back(i);
        if (missing.empty()) { error.clear(); return true; }
        Bytes body; number(body, missing.size());
        for (auto i : missing) string(body, keys[i]);
        try {
            const auto response = request('B', unit, body);
            size_t at = 0;
            for (auto i : missing) {
                if (response.size()-at < 8) throw std::runtime_error("truncated Mooncake batch");
                uint64_t n = 0;
                for (int b = 0; b < 8; ++b) n |= uint64_t(response[at++]) << (b*8);
                if (n > response.size()-at) throw std::runtime_error("invalid Mooncake batch size");
                data[i].assign(response.begin()+at, response.begin()+at+size_t(n)); at += size_t(n);
            }
            if (at != response.size()) throw std::runtime_error("trailing Mooncake batch bytes");
            error.clear(); return true;
        } catch (const std::exception& e) { error = e.what(); data.clear(); return false; }
    }
    bool erase(const std::string& key, std::string& error) override {
        // A local eviction cannot delete a shared object's other users. The
        // Mooncake unit policy owns remote reclamation and observes reader leases.
        return !local_ || local_->erase(key, error);
    }
    std::vector<std::string> manifests(std::string& error) override { return local_ ? local_->manifests(error) : std::vector<std::string>{}; }
    void prune(const std::vector<std::string>& live, std::string& error) override { if (local_) local_->prune(live, error); }
    std::vector<std::string> lookup(const Prefix& prompt, std::string& error) override {
        try {
            auto result = request('L', "", encode(prompt));
            std::vector<std::string> keys;
            std::string key;
            for (auto c : result) { if (c == '\n') { if (!key.empty()) keys.push_back(key); key.clear(); } else key += char(c); }
            if (!key.empty()) keys.push_back(std::move(key));
            return keys;
        } catch (const std::exception& e) { error = e.what(); return {}; }
    }
    bool announce(const Publication& p, std::string& error) override {
        Bytes b = encode(p.prefix); string(b, p.parent);
        string(b, std::to_string(p.value)); number(b, p.refs.size());
        for (const auto& key : p.refs) string(b, key);
        try { request('A', p.key, b); return true; }
        catch (const std::exception& e) { error = e.what(); return false; }
    }
    void touch(const std::string& key) override { request('T', key, {}); }
    void dead(const std::string& key) override { request('D', key, {}); }
    void invalidate(const std::string& key) override { request('X', key, {}); }
};
}
std::shared_ptr<Storage> mooncake_storage(std::shared_ptr<Storage> local) { return std::make_shared<Mooncake>(std::move(local)); }
}
