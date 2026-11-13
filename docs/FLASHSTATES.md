# Conversation state cache

`MAYA_STATE_CACHE=1` enables the shared conversation-state manager. It retains
prompt checkpoints and completed answers, including recurrent, attention,
convolution, indexer/compressor and required draft context. A later request can
resume the deepest exact matching prefix after another conversation ran, or
after the engine restarted.

```text
native sequence arena → bounded transfer buffers → RAM / SSD / Mooncake
                                     complete manifest published last
```

The native arena remains private while kernels write it. Retained chunks are
immutable and deduplicated by SHA256. A partial attention tail becomes a new
chunk on continuation; unchanged chunks stay shared. Backends requiring a
contiguous arena materialize those chunks before resuming. The RAM/SSD counters
describe retained payloads; they exclude the model's already allocated live
arenas, allocator overhead, token metadata and transfer scratch.

The feature starts disabled. The same controls work in the config's `env`
object or through `./maya.sh --setup --env MAYA_STATE_CACHE=1`.

Measured benefits, remaining costs and the enabled-main comparison are in
[the performance report](FLASHSTATES-PERFORMANCE.md).

## Controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAYA_STATE_CACHE` | `0` | Enable complete conversation states |
| `MAYA_STATE_RAM_GB` | `2` | Retained immutable host payload budget, GiB |
| `MAYA_STATE_DISK_GB` | `16` | Local retained SSD payload budget, GiB |
| `MAYA_STATE_MAX` | `128` | Maximum retained recovery units |
| `MAYA_STATE_DAYS` | `7` | Maximum age of persistent units imported at startup; uses refresh the manifest timestamp |
| `MAYA_STATE_MIGRATE` | `1` | Import compatible kept legacy text slots through the GLM/DeepSeek native adapter |
| `MAYA_STATE_DIR` | pack's `state-cache`, or Metal GGUF's `.state-cache` sibling | Private persistent directory; one engine owns it at a time |
| `MAYA_STATE_WRITE_THROUGH` | `1` | Queue completed captures to storage immediately |
| `MAYA_STATE_STAGING_MB` | `256` | Interior export scratch ceiling, MiB; the separate shared transfer pool uses at most `min(8, ceiling)` MiB |
| `MAYA_STATE_LEAF_LAST` | `1` | Evict dead states, then interior states, then live leaves |
| `MAYA_STATE_IDLE_SECONDS` | `60` | Demote an unused leaf; `0` disables idle demotion |
| `MAYA_STATE_ANCHOR_TIER` | `1` | Protect explicit anchors and shared branch points |
| `MAYA_STATE_SUPERSEDE_LAG` | `2` | Previous completed frontiers retained for rewritten answers; intermediate checkpoints and branches remain eligible independently |
| `MAYA_STATE_COST_D`, `MAYA_STATE_COST_E` | calibrated | Prefill seconds: `D*N + E*N*N/200000` |
| `MAYA_STATE_COST_CALIBRATE` | `1`, or `0` with explicit D/E | Fit nonnegative coefficients from up to 16 observed prefill intervals |
| `MAYA_STATE_DETOK_PRIME` | follows cache switch | Prime streaming detokenization with the last 16 prompt tokens, suppressing their text |
| `MAYA_STATE_BACKEND` | local | Set `mooncake` to add the shared tier |

A zero unit limit rejects captures. A state larger than the RAM budget streams
directly into the available SSD or shared tier. RAM can be zero when storage is
enabled. Capture hashes bounded chunks, admits the exact deduplicated physical
bytes, then transfers them while the native allocation remains owned. Large
loads use bounded batches and never materialize the full state in RAM. Active readers and pending
writes pin complete units; pressure can reject a new capture. It never evicts a
reader's required chunks. An asynchronous SSD write failure leaves an already captured RAM state usable;
a failed direct spill rejects that capture without publishing a manifest. Both
appear in diagnostics. A missing, mismatched or corrupt component makes the whole load
fail; native adapters reset any partly restored engine state before recomputing.

An identical prompt needs a newly computed sampling row. The lookup leaves at
least one actual token to forward unless a payload explicitly contains valid
sampling logits. Recurrent checkpoints keep their real endpoint; changing a
counter cannot rewind recurrence.

## Engine paths

| Engine | Retained state and endpoint handling |
| --- | --- |
| GLM CUDA/HIP | KDA recurrence/conv, MLA latent rows, completed indexer pools and open pool, positional GLM-DSA state, NextN endpoint rows and DFlash ring context; KDA side outputs preserve an interior prefill point |
| Qwen CUDA/HIP | GDN/conv, PLE history, indexer tails/pooled rows, every QSA encoding and page mapping, window-qualified MTP context and target residual; GDN/PLE/indexer side outputs assemble complete pipeline checkpoints |
| DeepSeek CUDA/HIP | Window rings, partial compressor groups, compressed KV/index sources, token/Engram history and every TP member; a stop inside accepted drafts reconstructs the consumed endpoint from an actual checkpoint |
| MiniMax CUDA/HIP | Written KV codes and scales, sparse index keys, authoritative streamed backing and DSpark context; layer and TP layouts have separate fingerprints |
| GLM Metal | KDA/conv, MLA/indexer state, endpoint hidden state, NextN context and DFlash ring; speculative acceptance replays projections from one initial recurrence |
| MiniMax Metal | Written BF16 KV and sparse index prefix |

CUDA GLM and Qwen already use compact speculative recurrence: one canonical
state plus inputs for accepted-update replay. The Metal GLM path now uses the
same memory principle. Publication waits for the actual committed native state.
Metal and unsupported exporter geometries use exact prefill cuts. They do not
advertise an interior state that a kernel did not produce.

Legacy slot formats are not interchangeable with a complete unit. GLM and
DeepSeek can import compatible kept text slots at startup, honoring their old
keep switch and age limit. The native loader validates the original layout,
then the adapter publishes complete units to the new manager. Source files are
retained. Multimodal legacy slots lack the new embedding identity and are
recomputed. Existing cheap native rewind checkpoints still support the active
sequence. Explicit old slot byte/count/age settings are translated with a
diagnostic; `MAYA_STATE_*` settings take precedence. Cache directories and
manifests are versioned independently of model weights; incompatible versions
safely miss.

## Optional Mooncake

The connector targets FutureMLS-Lab/Mooncake at
`59ca03f3c44131cf0ecea299d05bb8ea31f2032f`. Install that fork's `mooncake.store`
Python extension in the interpreter selected below. Ordinary local builds need
no Mooncake dependency. On Unix the native manager starts one private helper
process, exchanges bounded binary messages over a private socket, and keeps
model data out of JSON. Python's client performs the real TCP/RDMA store calls;
HTTP is used only for recovery-unit policy updates.

With Mooncake's system build dependencies installed, configure Maya with
`-DMAYA_BUILD_MOONCAKE_CLIENT=ON`, then build the optional
`maya-mooncake-client` target. It pins the source and pybind11 submodule and
places the TCP client and master under `build/mooncake-client/runtime`.
Set `PYTHONPATH` to its `python` directory and use the matching interpreter.
The target does not install system packages or start a service.

Start a dedicated master with `--eviction_policy=unit_gds`. The master's relevant
settings are `unit_leaf_last`, `unit_idle_demote_s`, `unit_orphan_grace_s`,
`unit_anchor_tier` and `unit_dead_headroom_ratio`. Its unit policy is incompatible
with HA, snapshots, internal offload, CXL and multiple tenants; the pinned master
rejects those combinations. Maya's separate local SSD cache remains supported.

```sh
export MAYA_STATE_CACHE=1
export MAYA_STATE_BACKEND=mooncake
export MAYA_STATE_MOONCAKE_MASTER=127.0.0.1:50051
export MAYA_STATE_MOONCAKE_ADMIN=http://127.0.0.1:9003
export MAYA_STATE_MOONCAKE_NAMESPACE=my-model-v1
export MAYA_STATE_MOONCAKE_HOSTNAME=127.0.0.1
export MAYA_STATE_DIR=/path/private-to-this-engine
```

| Optional variable | Default |
| --- | --- |
| `MAYA_STATE_MOONCAKE_PYTHON` | `python3`, with the pinned extension installed |
| `MAYA_STATE_MOONCAKE_HELPER` | `tools/state_mooncake.py` in the source tree used to build Maya; set an installed path when relocating |
| `MAYA_STATE_MOONCAKE_METADATA` | `P2PHANDSHAKE` |
| `MAYA_STATE_MOONCAKE_PROTOCOL` | `tcp`; `rdma` needs the corresponding client/runtime |
| `MAYA_STATE_MOONCAKE_RDMA` | empty device list |
| `MAYA_STATE_MOONCAKE_SEGMENT_MB` | `64` MiB contributed by each helper; size for the actual deployment |
| `MAYA_STATE_MOONCAKE_TIMEOUT_MS` | `10000`, bounded to 100–60000 |
| `MAYA_STATE_MOONCAKE_WAIT_MS` | `0`; optional missing-frontier wait, bounded to 1000 ms |

Two engines use the same namespace, artifact/layout fingerprint and tokens, with
different local directories. The connector publishes content chunks and the
complete manifest before a locator. Locators retain four deepest entries per
hash unit. Readers probe at most 4096 hash units, validate the exact native
prefix and every required object, and batch-load only chunks absent locally. Metadata
probes are memoized briefly; member existence is always rechecked to renew the
real store leases. A concurrent locator rewrite can lose a hit, never authorize
a different prefix.

Each manifest is its unit's exclusive member; content chunks and its locator
are shared refs. Units use absolute prefill cost and explicit parent hints.
Content equality does not imply token alignment, so the connector registers
these units as unaligned. Unit updates follow the bounded native write queue;
touches renew only the chosen state, and delayed supersession sends dead marks.
Local capacity eviction never deletes shared payloads. Explicit numerical/native
invalidation removes the affected manifest so locator memos cannot select it again;
shared chunks remain available to their other units. Load failure
recomputes. Startup republishes live local metadata; failed touches queue a
repair publication.

## Diagnostics and checks

Native `STATE` lines, `/metrics.state_cache`, Prometheus
`maya:state_cache_*` readings and Monitor's conversation-cache card are separate
from the expert-weight cache. They include lookup hits by tier, confirmed native
restored tokens, physical RAM/SSD payload bytes, logical/unique/shared bytes,
live pins, admissions/rejections, eviction/demotion/promotion, queued writes,
storage errors and fitted cost coefficients. A lookup hit can be superseded by
a deeper live native prefix without loading it; confirmed restored tokens count
only adapters that completed every device copy.

`tools/agent_bench.py run --answer-fed ...` feeds real answers into the next
request. The default scripted mode remains reproducible independently of model
output. Use the same mode on both sides of a comparison.

The focused native checks are `state_cache_test`, `state_io_test`,
`glm_checkpoint_export_test`, `gdn_checkpoint_export_test`,
`glm_state_cache_test PACK`, `dflash_state_cache_test DFLASH_GGUF` and
`metal_state_cache_test GLM_GGUF MM3_GGUF`.
`tools/glm_state_fixture.py --out FILE` generates a CUDA-compatible synthetic
GLM source; `--mtp` includes NextN. Pack it with `tools/iq_pack.py`.
`tools/test_native_state_cache.py --engine BINARY --pack PACK` exercises the
normal native protocol, displaced parallel contexts and legacy migration. Add
`--ram-gb 0.000001 --skip-migration` to force bounded spills; legacy import has
its own host-buffer limit. Generate the small DFlash test artifact with
`tools/glm_dflash_fixture.py --native-small --out FILE`. `tools/test_state_mooncake_live.py`
starts an isolated real master and checks two native engine instances with an
MTP fixture. It requires the pinned client and accepts explicit master, engine
and model paths. CPU graph validation does not establish Metal GPU parity.
