# FlashStates validation

Observed on Linux with CUDA 12.4 and RTX 3090 (SM86), using synthetic model weights.
These are correctness checks; they do not establish a workload speedup.

| Boundary | Observed result |
| --- | --- |
| Shared manager | Strict C++20 build, behavioral checks, ASan and UBSan pass; real bytes, dependency pressure, leases, restart, corruption, failed copies, supersession, batched loads, imported RAM-state revocation and bounded capture beyond RAM |
| GPU transfers | Overlaid views, four QSA formats, 19 MiB transfer with a 1 MiB staging ceiling, streamed SSD restore and zero Compute Sanitizer errors |
| Interior exports | KDA modes 0/4/8 and serial/split GDN exports preserve full-call outputs/state bitwise; exported recurrence, convolution, PLE and indexer state match their exact prefix |
| CUDA GLM | Full-vocabulary logits match bitwise after an intervening request and SSD restore; interior export leaves the full call unchanged; native protocol checks retry, answer continuation, displaced parallel contexts and read-only legacy migration |
| NextN boundary | An aligned interior checkpoint restores the same open-pool values as the exact-endpoint reference, with matching microbatch shapes; the old endpoint fails the regression |
| CUDA speculation | With approximately 1 KiB retained RAM, native MTP offered 48 drafts and native DFlash offered 196 drafts; completed-answer continuation and identical-prompt retry match ordinary decoding after displacement and restart |
| DFlash context | Wrapped ring overwritten by another context restores written keys, values and tags exactly from SSD; Compute Sanitizer reports zero errors |
| Metal graph oracle | GLM and MiniMax full-state restores pass after overwritten context; malformed units are rejected; compact recurrent speculation matches 40 ordinary tokens over 20 windows with zero logit difference |
| Real Mooncake | Pinned master/client and two native Metal CPU engines pass speculative completion, retry, continuation, missing-member fallback, master restart, SSD metadata repair and explicit invalidation; RAM is smaller than the saved unit; installed client/master run without build-directory library paths |
| Tools/server | Tools suite: 921 tests, 40 skipped. Server suite: 953 tests, 40 skipped, pass on the final rerun; an earlier unchanged MCP crash-stderr timing failure also passed three focused reruns. State diagnostics and detokenizer checks pass |
| Build coverage | CUDA GLM/Qwen, DeepSeek and MiniMax binaries build; the HIP CUDA-name compatibility check passes. CI includes Windows persistence and macOS Metal shader/adaptor compilation plus CPU oracle checks |

The synthetic CUDA tests exercise one SM86 device. Native Qwen and MiniMax
model generation, other CUDA architectures, HIP hardware, Windows GPU execution
and Metal GPU arithmetic were not exercised here. Build coverage and the Metal CPU
oracle do not substitute for those hardware checks. Local and optional shared
storage are opt-in through `MAYA_STATE_CACHE=1`; see [controls and reproduction
tools](FLASHSTATES.md).

Real GLM and DeepSeek model loading, exact native-byte probes and the completed
65-trial performance study are documented in [the performance report](FLASHSTATES-PERFORMANCE.md).
That study does not extend generation coverage to the other backends listed above.
