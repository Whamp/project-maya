#pragma once

#include "strata/core/session.hpp"
#include "strata/core/state_io.hpp"

namespace strata::core {
// Appends a canonical logical view of all written attention cells. Streamed
// caches use their authoritative host copy; resident caches honor page_table.
bool qsa_state_regions(const ModelGeometry& g, QsaState& state, int64_t endpoint,
                       const std::string& name, bool indexer,
                       std::vector<statecache::MemoryRegion>& memory, std::string& error, int64_t first_cell = 0);
void session_running_regions(const ModelGeometry& g, SessionState& state,
                             const std::string& name, std::vector<statecache::MemoryRegion>& memory);
bool qsa_state_loaded(const ModelGeometry& g, QsaState& state, int64_t endpoint, std::string& error);
}  // namespace strata::core
