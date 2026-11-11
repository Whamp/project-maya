# FlashStates implementation plan for Project Maya

Implement the complete set of FlashStates state-management techniques inside Maya's native engines. The result is exact conversation reuse at prompt, answer, and interior-prefill positions, managed as complete resumable states across GPU memory, RAM, SSD, and an optional shared Mooncake store.

This is an implementation plan. All milestones below are in scope. Measurements are delivery and regression checks, not a preliminary trial or a requirement to justify starting the work.

Maya keeps its model formats, expert offloading, inference engines, and external APIs. The integration ports the techniques and their required storage mechanisms; it does not require moving inference to vLLM.

## Scope at a glance

1. **Save exact states:** preserve the processed endpoint of every completed turn, plus useful earlier restore points.
2. **Capture inside prefill:** export recurrent state without ending the surrounding prefill call.
3. **Keep speculation compact:** maintain one canonical running recurrent state and commit only accepted updates.
4. **Manage states together:** admit and evict complete recovery units, accounting for their shared prefix pages and recomputation cost.
5. **Move them between tiers:** bounded asynchronous GPU/RAM/SSD transfers and an optional shared Mooncake backend.
6. **Finish the integration:** prompt-aware streaming detokenization, state-copy and dtype safeguards, configuration, diagnostics, and tests across Maya's engines.

## Reference versions

Planning baseline: Maya `0065548`, v1.1.57. Rebase implementation against the then-current default branch and refresh these touchpoints before changing code.

- [FlashStates release repository](https://github.com/FutureMLS-Lab/FlashStates/tree/e2ea9a7864b41081a0147a3968e73bd5c6875a7d).
- [Release paper](https://github.com/FutureMLS-Lab/FlashStates/blob/e2ea9a7864b41081a0147a3968e73bd5c6875a7d/docs/FlashStates_Release.pdf), especially sections 3–4 and appendices G, H, and J.
- [Complete feature switches and accompanying fixes](https://github.com/FutureMLS-Lab/FlashStates/blob/e2ea9a7864b41081a0147a3968e73bd5c6875a7d/docs/what-flashstate-changes.md).
- [vLLM checkpoint implementation](https://github.com/FutureMLS-Lab/vllm/tree/3cfd432cd83a54a5e538cad7b090e56c27a870f9/vllm/v1/core/hybrid_checkpoint).
- [Mooncake frontier transfer protocol](https://github.com/FutureMLS-Lab/vllm/blob/3cfd432cd83a54a5e538cad7b090e56c27a870f9/docs/design/mooncake_store_frontier_checkpoints.md).
- [Mooncake recovery-unit policy](https://github.com/FutureMLS-Lab/Mooncake/blob/59ca03f3c44131cf0ecea299d05bb8ea31f2032f/docs/source/design/store/unit-eviction-policy.md).

The paper and released code differ in some exporter coverage. Use the pinned code for implementation details and the paper for the algorithm's contract. Preserve applicable upstream licenses and notices for copied implementation or kernels.

## Complete feature map

| FlashStates technique | Maya implementation | Milestone |
| --- | --- | --- |
| Arbitrary state checkpointing and exact request-end capture | Publish a complete state at its actual processed token count, including partial attention tails | 2 |
| Prompt-end and shared-prefix restore points | Represent system prompts, turn boundaries, branches, and prompt tails in the same state index | 2 |
| In-chunk recurrent export | Add side-output checkpoints to KDA and GDN, plus required auxiliary-state capture | 3 |
| Export alignment and filler planning | Capability-aware prefill planning with aligned export, bounded partial chunks, and exact fallback cuts | 3 |
| Compact speculative recurrent state | One canonical recurrent state per sequence; fold accepted updates after verification | 4 |
| Exact speculative endpoint and stop handling | Track verified, accepted, emitted, and actually processed tokens separately | 4 |
| Recovery units and prefix dependency graph | Couple each checkpoint with every attention/auxiliary allocation needed to resume it | 1–2 |
| Shared-prefix storage and private mutable tails | Reference-count immutable prefix chunks; copy on write when a continuation modifies a tail | 1–2 |
| Prefill cost model | Fit per-model/runtime coefficients using Maya's actual offload configuration | 5 |
| GDS admission and eviction | One dependency-aware priority for both retaining and evicting states | 5 |
| Leaf-last, branch-point protection, idle demotion | Explicit tiers and clock semantics for the prefix graph | 5 |
| Dead/incomplete-state and orphan reclamation | Reclaim unusable allocations first, preserve live-request leases | 5–7 |
| Host write-through of completed states | Queue complete state exports before request allocations are released | 6 |
| Newest-state lookup and local-page reuse | Locate the deepest complete matching state and transfer only missing chunks | 6–7 |
| Supersession with lag and rewrite anchors | Retire obsolete states while retaining earlier states needed for retries and rewritten turns | 5–7 |
| Bounded waits and lookup memoization | Coordinate in-flight writes, reuse lookup metadata only within lease validity | 6–7 |
| Pinned asynchronous GPU/host staging | Bounded staging buffers, separate copy/storage work, event-based allocation release | 6 |
| Mooncake recovery-unit eviction | Optional connector registering members, shared refs, costs, parents, touches, and dead units | 7 |
| Mooncake anchor tier and dead headroom | Configure and verify the pinned fork's complete unit policy | 7 |
| Prompt-suffix detokenizer priming | Initialize streaming decode with prompt context without emitting that context | 8 |
| Unwritten state-column safeguard | Publish only state components confirmed written at the advertised endpoint | 1, 4, 8 |
| BF16 MTP loading safeguard | Preserve artifact dtype semantics when loading MTP weights and constructing execution views | 8 |
| Overlaid KV-view copy safeguard | Enumerate and copy all required views once, preserving alias relationships | 1, 8 |
| Feature configuration and observability | Unified controls, capability reporting, cache counters, diagnostics, and reproducible request traces | 9 |

## Architecture

### One shared native manager

Add a CPU-testable `strata::core::StateCache` module. Place its public interface in `include/strata/core/state_cache.hpp` and its implementation under `src/core/state_cache/`. Link it into each native engine rather than duplicating policy inside their request loops.

The manager owns exact prefix lookup, recovery-unit dependencies, admission, eviction, leases, tier placement, and storage jobs. Engine adapters own the interpretation and movement of model state. KDA, GDN, sparse attention, and compressed attention keep their existing computation kernels except where checkpoint export or compact state requires changes.

Keep the calling interface small:

- `find_resume(prefix)` returns a leased restore plan and the exact reusable length, or a miss.
- `publish(captured_state)` registers a fully materialized checkpoint and applies admission.
- `release(lease)` ends a request's hold; the manager advances pending transfers and reclamation at scheduler safe points.

Use explicit capture/restore operations at the engine adapter seam. Storage adapters perform lookup, asynchronous write/read, and lease release. Policy stays independent of CUDA, HIP, Metal, and Mooncake headers.

### Data and ownership

| Type | Required meaning |
| --- | --- |
| `PrefixIdentity` | Exact token-prefix identity, model/artifact fingerprint, state-layout version, relevant runtime settings, multimodal inputs and positions, and configured cache namespace |
| `StateAllocation` | An owned allocation or view, its physical bytes, tier, valid token range, layer/group/device identity, completion event, and generation |
| `CapturedState` | The processed token count and complete set of attention, recurrent, convolution, indexer/compressor, token-history, and required draft-carrier state at that count |
| `RecoveryUnit` | A captured prefix, owned state members, shared allocation references, earlier fallback, continuations, use stamp, and policy tier |
| `StateLease` | A hold protecting every allocation a running/queued request or transfer still needs |
| `RestorePlan` | The selected complete unit, exact restored endpoint, locally available chunks, missing chunks by tier, completion events, and whether valid sampling logits are available |

State publication follows `capturing → ready → published`. Loading follows `leased → loading → ready for engine use`. Failure never produces a partially published unit or an engine state treated as successfully restored.

Use explicit owned handles and completion events. A pointer into a reusable sequence arena alone is not a retained state. Transfer jobs hold allocations until their copies finish, including after cancellation or sequence reassignment.

Sampling RNG and response-parser state remain request-private. Prefix lookup validates the exact prefix and its extra input identity; a hash match alone is not sufficient authority to restore a state.

### Shared storage over Maya's existing arenas

Maya currently uses large per-sequence arenas in several engines. Refcounting metadata alone would not create FlashStates' shared storage or correct reclaimed-byte accounting.

Implement an immutable chunk store for per-position attention/cache regions, with exact valid lengths and copy-on-write partial tails. Common prefixes reference the same physical retained chunks. Recurrent and auxiliary state are separate owned members of their recovery unit. Avoid serializing the whole conversation again for each checkpoint.

Where kernels already accept mapped/page-backed storage, attach compatible read-only chunks. Where kernels require contiguous arenas, the adapter materializes the required chunks into the sequence's private arena. Charge these live copies separately and never claim they are shared physical memory. Native allocation leases may retain an idle sequence arena without copying until that sequence is reassigned; detachment then preserves its retained chunks before reuse.

This supplies genuine shared-prefix retention without making a rewrite of every attention kernel a prerequisite. GPU cache, live arenas, private tails, verifier scratch, transfer staging, RAM cache, and SSD usage must all have explicit accounting.

Reserve copy destinations before starting transfers. Coordinate cache reservations with Maya's expert-pool sizing at scheduler safe points; avoid unbounded allocations that steal the headroom used by expert offloading.

### Model adapters

| Engine | Capture and restore contents | Existing touchpoints |
| --- | --- | --- |
| GLM-5.3-Flash | KDA matrix state and convolution history; DSA latents, indexer key/gate rings and partial pools; MTP carrier where required | `glm_model.hpp`, `glm_state_layout.hpp`, `glm_fast_path.cu`, `glm_prefill.cu`, `generate.cpp` |
| GLM-5.3 positional DSA | Latents, rotary-key tails, index keys, exact valid ranges, MTP carrier; no invented KDA state | `glm_prompt_cache*.cpp/.cu`, `glm_state_layout.hpp` |
| Qwen hybrid | GDN recurrence/convolution, QSA KV and indexer tails, PLE history/token window, split-stage state, drafter state as required | `session.hpp`, `session.cpp`, `ConvCheckpoint` and generic request loop in `generate.cpp` |
| DeepSeek V4.1 | Attention windows, compressor partial groups, compressed KV and index keys, Engram/token history, restore points, relevant DSpark state | `src/ds4/ds4_model.hpp/.cu`, `ds4_main.cpp` |
| MiniMax M3.1 | Per-sequence sparse-attention caches, indexer state and valid positions, multimodal identity, relevant DSpark state | `src/mm3/mm3_model.hpp/.cu`, `mm3_main.cpp`, `mm3_slots.hpp` |
| Metal engines | The corresponding model state through ggml/Metal storage and completion mechanisms | `src/metal/glm_metal*`, `main.cpp`, and the existing MiniMax Metal path |

Inspect each adapter's draft state to distinguish state required for target-model equivalence from scratch that can be reset or reconstructed. Positional architectures use the same manager and policy without adding recurrent fields they do not have. Apply the techniques to Maya's existing supported models; introducing unrelated model architectures is separate model-support work.

## Implementation milestones

### 1. Establish recovery units and allocation ownership

Build the shared types, exact prefix index, immutable chunks, reference accounting, leases, and adapter interface. Derive region descriptors from existing model layouts instead of maintaining another handwritten offset list. Record all aliased views and valid ranges.

Extract state-retention responsibilities from GLM's request-loop lambdas and Qwen's `ConvCheckpoint` plumbing. Add adapters for DeepSeek, MiniMax, and Metal. Their current save/restore primitives are starting points, not a second policy layer around the new manager.

**Completion:** every adapter can describe, retain, restore, and release a complete state; shared chunks remain alive until their final holder releases them; sequence arena reuse cannot corrupt a retained state. Tests cover actual multi-view payloads, refcounts, failed capture, stale generations, and outstanding copies.

### 2. Publish exact prompt and answer endpoints

Publish at prompt completion, normal answer completion, useful shared-prefix boundaries, and existing retry/restore points. GLM's normal finish currently leaves the reusable snapshot near the previous prompt's end; replace that limitation with a retained processed-answer frontier. Keep an earlier compatible restore point for retries, edits, and cancelled-answer replacement.

Track consumed tokens independently from emitted tokens. If the final sampled token has not been forwarded, advertise the actual processed position and leave that token for the next request. Never claim to have processed it. Preserve Maya's useful cancellation recovery, publishing only confirmed complete state, even though the reference implementation skips aborted-request publication.

Match the longest complete exact prefix. Include image identity, reasoning/tool token history, and runtime fingerprints. A divergent prompt selects an earlier valid unit. Give a continuation private recurrent/auxiliary state and a copy-on-write mutable attention tail.

When generation needs its first sample, a restore must leave at least one prompt token to forward unless the adapter also restores valid endpoint logits or sufficient output-head input to reconstruct them. Retain Maya's pre-final-token prompt checkpoint for this purpose. For an identical-prompt retry without restored logits, choose a complete earlier checkpoint and replay the remaining suffix. A recurrent checkpoint cannot be rewound by merely decrementing its recorded endpoint. A positional adapter may truncate only when it can reconstruct every required component at the shorter endpoint. Full-prompt reuse is allowed only when the output-head readiness contract is satisfied.

**Completion:** normal turns reuse their retained processed answer, retries restore earlier states, and edited branches preserve siblings. Tests cover block/chunk boundaries, tiny prompts, output-budget stops, multiple contexts, images, and corrupt or missing components. Include an identical-prompt retry after another request has replaced the live logits, both from a local checkpoint and after a tier restore.

### 3. Export checkpoints from inside prefill

Add an export descriptor carrying a requested absolute position and output destinations. The recurrence writes the interior checkpoint as a side output while producing its normal outputs and final live state. Capture convolution history, indexer tails/partial pools, and PLE state at the same endpoint; exporting the recurrence matrix alone is insufficient.

Implement KDA export in Maya's own recurrence and split-recurrence kernels, then GDN export in its native/fused paths. Both implementations preserve the original full-call computation. Cover CUDA, HIP, and the corresponding Metal recurrence operations. Positional DSA uses its existing earlier-position capture rather than a recurrence exporter.

The active KDA prefill entry points are `include/strata/kernels/glm_batch.hpp` and `src/kernels/cuda/glm_batch.cu`, called from `glm_prefill.cu`; the separate reference/token paths include `glm_kda.cu` and `glm_fast.cu`. Qwen's GDN work starts from `session.cpp`, `gdn.cu`, `native_gdn.cu`, and `fused_gdn.cu`. Extend the dispatched paths as well as the parity/reference paths.

Expose per-backend supported offsets, alignments, and export count. Start with one interior export per sequence per call, matching FlashStates' contract. The planner passes a supported mark without draining the full prefill pipeline, ends at a second required mark, and uses an exact cut when an offset is unsupported. Include aligned filler chunks when strict checkpoint alignment would otherwise leave available scheduling capacity unused. Use Maya's own shape constraints rather than copying vLLM's block sizes.

**Completion:** export-on and export-off runs with the same full-call shape produce the same live outputs/state under that backend's numerical contract. Exported state restores to the requested endpoint. Tests exercise offsets before/at/inside tiles, short convolution windows, multiple marks, split pipelines, graph replay, and fallback paths. Unsupported offsets remain exact through deliberate cuts, with a visible reason.

### 4. Compact speculative state and exact commit

Unify request commit bookkeeping across ordinary decode, MTP, DFlash, DSpark, and lookup drafting. Record verification start, candidates processed, accepted prefix, stop position, emitted tokens, and actual canonical-state endpoint. Publish after the corresponding completion events, before releasing or rebinding storage.

Use one canonical persistent recurrent state per active sequence. Verifier scratch may retain bounded temporary information; accepted-update replay or folding replaces persistent full-state copies for every speculative column. Restore convolution/indexer/carrier state consistently. Truncate acceptance at EOS, client stop, thinking-budget, and cancellation boundaries. Reset or rewind drafter state when appropriate.

Maya uses pipeline and multi-sequence paths that the reference implementation restricts. Implement generation-tagged commit/completion tracking per request and stage; do not copy those restrictions into a silent feature disable. Where a path cannot yet expose a valid state, synchronously finish or reconstruct its committed state before publication.

**Completion:** all acceptance lengths, total rejection, stops inside an accepted window, asynchronous finishes, and interleaved sequences preserve target-model state. Exercise existing speculative rollback tests and add finish/restore tests for each draft mode. No store reader can observe uncommitted columns.

### 5. Implement dependency-aware GDS admission and eviction

Use the same policy for retaining a newly captured state and choosing victims under pressure. With no displacement needed, admit a complete state. Under pressure, compare its priority with eligible retained states; rejecting cache admission never rejects the inference request.

Implement the prefix graph, deepest usable fallback, reverse allocation-to-holder index, lazily refreshed priority queue, and GDS watermark:

```text
gain(unit)  = estimated_prefill(fallback_position, unit_position)
freed(unit) = bytes actually reclaimable if this unit leaves
score(unit) = stamp(unit) + gain(unit) / freed(unit)
```

Native allocations have unequal sizes, so use physical reclaimed bytes, not token length or nominal full-prefix size. Count shared chunks only when their final holder leaves. A unit with no freeable bytes is not a capacity victim. Removing a unit recalculates descendants' fallbacks and affected shared-byte counts without refreshing their use stamps.

Refresh a stamp on capture/admission, actual reuse, and the final live holder's release. Implement dead/interior/leaf ordering, branch-point protection, idle-leaf demotion, rewrite-anchor protection, and deterministic tie-breaking. Match the reference's watermark behavior when leaf-last ordering is enabled. A shared prefix's stamp does not refresh merely because a descendant is reused.

Fit the reference model `C(a,b) = D*(b-a) + E*(b*b-a*a)/200000` using Maya prefill timings. Key coefficients by model, backend, device split, precision, and expert-offload configuration, since those change the cost. Keep explicit coefficient overrides and a documented conservative initializer so configuration does not require a separate tuning exercise. Calibration affects ranking, never state correctness.

Apply budgets separately per tier. Reclaim orphans, dead units, and incomplete units before useful states; retain active request/transfer pins. If the budget cannot accommodate another active request, use the scheduler's admission/backpressure path rather than evicting its dependencies.

**Completion:** deterministic tests cover incoming-state rejection, shared-prefix forks, fallback removal, zero-free-byte units, pinned requests, idle demotion, watermark updates, dead/orphan reclamation, and exact byte accounting. Sequence tests check consistency after repeated publish/use/evict operations.

### 6. Add asynchronous native RAM and SSD storage

Write completed frontiers through to the configured host tier, not only when another conversation displaces them. Store attention chunks once and separate checkpoint manifests from payloads. Preserve intermediate checkpoints when a GLM context is parked, addressing the current loss of those restore points on context displacement.

Use a bounded pinned staging pool and copy-completion events on CUDA/HIP. Free retained GPU allocations once their staging copy is complete; storage work proceeds independently. Use corresponding completion-aware staging for Metal. Keep ordinary host-cache capacity separate from scarce pinned staging capacity and expert RAM.

Implement newest-complete-state lookup, local-chunk reuse, missing-chunk-only restore, delayed supersession, bounded waits for a just-finished write, and lookup memoization. Expiration or deletion can turn a hit into a miss; load success requires validating every required component. A failed load resets any partially modified engine state before recomputation.

Version manifests, preserve compatible model fingerprints, publish metadata after payload completion, validate payload sizes/checksums, and recover from incomplete writes. Import compatible legacy slots through the model adapter; refuse incompatible layouts. Keep existing restart persistence and age/budget controls through the migration.

**Completion:** finish, displace, resume, restart, and retry work across RAM and SSD with shared prefixes and intermediate checkpoints. Fault tests cover disk full, short reads, interrupted writes, queue pressure, cancelled transfers, and repeated shutdown. Allocations remain valid until their last I/O reader finishes.

### 7. Integrate the optional shared Mooncake tier

Add a real Mooncake storage adapter to the same manager. Pin the FlashStates Mooncake fork and expose it as an optional dependency/build target. Native local operation remains self-contained; enabling Mooncake adds shared recovery across engine instances.

Implement the complete frontier protocol: content-addressed full pages, partial tails, per-device recurrent/auxiliary members, locator records, deepest exact matching frontier, every-device verification, and batch loads. Publish a usable shared frontier only after all its required objects have completed. Concurrent locator updates may cause misses, never a false match.

Implement newest-state-first lookup, reuse of local pages, supersession removal with configurable lag, bounded write waits, memoized probes and re-verification within lease validity, pinned staging, and per-job resource release. Handle changed local hits and store/master restarts. Do not reuse a cached lookup beyond the store's lease window.

Register units through the master's batch unit interface, including `members`, shared `refs`, absolute prefill `value`, `parent`, and alignment. Send ordered `upsert`, `touch`, and `dead` updates through a bounded queue. Configure `unit_gds`, leaf-last, idle demotion, anchor tier, dead headroom, and orphan grace. Surface update failures and recover by re-registering live metadata; missing metadata must degrade to safe misses or orphan reclamation.

Preserve the pinned fork's compatibility limits: `unit_gds` does not support its HA, snapshot, internal offload, CXL, or multi-tenant modes. Maya's local SSD tier is separate from Mooncake's unsupported internal offload. Capability checks must reject incompatible combinations clearly. The initial direct GPU connector targets supported Linux/CUDA builds; provide host-staged transport for other Maya backends where the Mooncake client is supported, and keep local-tier feature coverage on all platforms.

**Completion:** two Maya engine instances reuse a completed state from each other, including under speculative decoding. Tests cover missing objects on one device, expired leases, out-of-order completion, duplicate updates, master restart, pressure eviction, and partial unit loss. No uncommitted speculative state reaches the store.

### 8. Port streaming and correctness safeguards

Extend `serve/server.py::Detokenizer` to accept prompt context. Seed the incremental decoder with the minimal prompt suffix required by the tokenizer while suppressing all prompt text. Apply the same handling when thinking/continuation logic restarts a decoder. Preserve reasoning/tool parsers, stop strings, logprobs, and streaming/non-streaming equivalence. The current raw-byte incremental path may already meet part of the contract; add context handling only where the tokenizer needs it, with tests for all supported tokenizer paths.

Make written-component validity part of capture and speculative commit. A checkpoint mark or column not actually produced cannot become a hit. Audit MTP artifact dtype handling in loaders and pack planning: BF16 source tensors must not inherit unrelated quantization metadata, while deliberate supported conversions remain explicit. Use the allocation/view descriptor from milestone 1 for copy and serialization so every overlaid KV view is preserved.

**Completion:** Unicode fragments, whitespace-sensitive tokenization, special tokens, tool calls, reasoning resumes, and stop sequences stream correctly. State-copy fixtures verify every required alias/view, and loader fixtures cover mixed BF16 MTP and quantized trunk tensors. Fix the corresponding native invariant rather than transplanting vLLM-specific conditionals.

### 9. Wire configuration, diagnostics, and delivery checks

Expose a consistent `MAYA_STATE_*` configuration family for cache policy, tier byte budgets, exporter behavior, cost coefficients, leaf/idle/anchor settings, staging capacity, persistence, and optional Mooncake connection. Final names should follow existing setup/server conventions. Translate old per-engine settings into this configuration with deprecation diagnostics, then remove duplicated policy implementations as their callers migrate.

Add engine capability reporting and metrics for exact frontier hits, restored tokens by tier, export/cut positions, admission rejections, victim reasons, fallback depth, unique/shared/private bytes, live pins, transfer queues, lookup misses/failures, and skipped invalid captures. Make diagnostics distinguish state cache traffic from expert-weight offloading.

Update `docs/SETTINGS.md`, `docs/AGENT_BENCH.md`, setup/config handling, and the relevant Monitor cache summary. Keep API request/response shapes stable unless an existing metrics extension is required. Document model/backend capabilities and explicit fallback behavior.

Extend `tools/agent_bench.py` with an answer-fed continuation mode: its current scripted mode deliberately does not feed model answers into the next turn, so it cannot exercise completed-answer reuse. Preserve the scripted mode for scheduler comparisons. Add deterministic token-ID traces for exact-prefix behavior and normal API traces for tokenizer/template behavior.

**Completion:** the full feature set is exercised through Maya's normal server on every supported engine/backend combination, with clear diagnostics for architecture-specific paths and optional external transport availability. Build/install paths and the documented configuration work from clean environments.

## Delivery order and validation

The sequence is **ownership → exact capture → in-prefill export → speculative commit → admission/eviction → local tiers → shared tier → streaming/fixes → product integration**. The correctness safeguards begin with the data model and continue through each adapter; milestone 8 consolidates their cross-engine checks. All nine milestones are part of the implementation, not alternative proposals.

Use separate reviewable changes for the shared core, each engine adapter, exporters, speculative commit, policy, native tiers, Mooncake, and server integration. Run the relevant correctness checks at each step. Do not retain two independent managers for the same conversation state after its migration.

### Required checks

- **Core behavior:** real payload save/restore, exact prefix matching, graph/refcount invariants, capacity accounting, deterministic GDS decisions, leases, failures, and restart recovery.
- **Kernel behavior:** checkpoint contents at supported offsets, unchanged full-call outputs, split/graph ordering, and applicable CUDA sanitizer checks.
- **Generation behavior:** compare next-token logits and continuations with a fresh run under the same execution/numerical contract. Preserve bitwise equality where the backend promises it; record established tolerance for paths whose chunking changes arithmetic order. Include ordinary and speculative decode, retries, edits, cancellation, pictures, and tool/reasoning histories.
- **Storage behavior:** tier promotion/demotion, supersession, shared-page retention, in-flight reads/writes, all-device completion, partial failures, duplicate operations, and shutdown.
- **Server behavior:** run the existing engine-contract, multi-request, detokenizer, API, and model-specific tests together with the new state-manager cases.
- **Portability:** CUDA on representative older and newer supported architectures; HIP on supported RDNA/CDNA paths; Windows native storage; Metal on existing supported engine paths. Compile-time availability and runtime kernel selection must agree.
- **End-to-end characterization:** cold starts, long repeated turns, long answers, shared system prompts, branches, context edits, cancellation, and more sessions than fit in the cache. Record next-turn TTFT, output latency/throughput, replayed tokens, cache and staging bytes, expert-tier contention, and restore/write overhead. These checks characterize the delivered implementation and catch regressions; they do not narrow the agreed feature scope.

Hardware-dependent values such as fitted coefficients, staging sizes, and runtime dispatch are collected during implementation on the target machines. The plan assumes no universal speedup and makes no hardware-purchase requirement.

## Definition of complete

Every row of the feature map has an implemented path or an explicit architecture-equivalent operation, tested through the native engine that uses it. Backend alignment fallbacks are exact and visible. All Maya engines use the shared state lifecycle and policy; checkpoint-bearing engines preserve their required recurrent or compressed state; positional engines use their own state layout. Local GPU/RAM/SSD operation and the optional shared Mooncake path are integrated. Streaming, speculation, persistence, and failure recovery preserve their existing user-visible contracts.
