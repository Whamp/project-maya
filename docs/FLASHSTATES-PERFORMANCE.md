# FlashStates performance on Maya

FlashStates helps when a conversation returns after a long answer, or when
several conversations compete for a small retained-state budget. It does not
make every request faster. Main's existing cache is enabled in every comparison
below.

On one identical GLM continuation prompt, the first token arrived in **0.772 s
instead of 5.000 s**, across five runs. Under DS4 cache pressure, complete traces
took **504.3 s instead of 581.0 s**. Roomy DS4 and cold GLM requests retain an
extra cost.

## What changes relative to main

- **Keep the answer endpoint.** In the GLM long-answer case, main forwards 4,161
  tokens on the next turn. This implementation restores the completed answer
  and forwards 65. Both engines leave actual work to produce sampling logits.
- **Share retained bytes across recovery points.** Immutable chunks and complete
  recovery units let the manager retain useful states under pressure. In this
  DS4 trace it restored 12–14 returning requests per run; main restored zero
  under the same explicit budgets. Main still uses its enabled legacy cache.
- **Reduce publication cost.** Eligible captures retain the first read of missing
  RAM chunks. Fresh captures of at least 4 MiB can hash those private chunks in
  up to 16 jobs, joined before admission. Native reads stay on the caller thread.
  Small captures, matching prefixes and insufficient headroom keep the serial
  path. Full identities, manifests, storage validation and restore checks remain.

The feature remains opt-in with `MAYA_STATE_CACHE=1`. See
[configuration and engine coverage](FLASHSTATES.md).

## Results

Measured on `server60`, Linux, four RTX 3090s and a 12-core / 24-thread
Threadripper PRO 3945WX. Both model configurations use a 1,048,576-token maximum
context. GLM loads `Maya-S-v2-IQ2_XXS`; DS4 loads the Maya-S pack of
`DeepSeek-V4.1-Flash-1017-GGUF` with four-way tensor parallelism and DSpark.
These results describe this host, models and recipes.

Main is engine snapshot `6fba6be981258d085c35fd99ef08c4131df49f95`, also the
PR target `flashstates-integration` at measurement. The optimized PR is
`3433c7d` plus the publication change in this PR, called **C3** in the raw
experiment. Its exact `state_cache.cpp` SHA-256 is
`ba8460c69aba00302a1667ae3390f01091d5a394e0cb8b94b88d7800d77a3740`.
Both rebuilt native executables are byte-for-byte identical to the measured
private C3 executables.

Each arm has five independent process runs. Odd repeats reverse arm order.
Except for the explicitly identified single prompt, each value is the median
of per-run medians; brackets contain the minimum and maximum of those five
per-run values. Ranges are not confidence intervals.

| Measurement, lower is better | Main, existing cache enabled | Optimized PR |
| --- | ---: | ---: |
| GLM short, returning-request first token | 955.7 [952.9–974.8] ms | 772.2 [768.7–785.6] ms |
| GLM after 4,096-token answer, **identical prompt**, agent 1 / turn 2 | 4,999.8 [4,991.1–5,010.7] ms | 772.5 [754.4–792.5] ms |
| DS4 roomy, returning-request first token | 1,972.1 [1,943.1–2,068.8] ms | 2,140.8 [2,115.7–2,261.3] ms |
| DS4 pressure, returning-request first token | 11,579.4 [11,552.8–11,614.6] ms | 9,177.0 [8,335.9–9,699.6] ms |
| GLM short, cold-request first token | 4,942.9 [4,933.4–4,949.7] ms | 5,124.7 [5,090.6–5,150.3] ms |

Whole-trace time starts at native READY and ends after QUIT, pending cache work
and native process exit. Model startup is recorded separately. These totals
include cold requests and all generated tokens.

| Complete trace | Main | Optimized PR |
| --- | ---: | ---: |
| GLM short | 39.250 [38.923–39.509] s | 39.285 [39.035–39.531] s |
| GLM long answer | 217.577 [217.412–217.832] s | 209.513 [209.342–209.942] s |
| DS4 roomy | 71.934 [71.147–74.650] s | 72.565 [71.564–76.314] s |
| DS4 pressure | 580.989 [580.600–586.141] s | 504.333 [495.316–507.334] s |

The short GLM trace is effectively level overall: its warm benefit offsets its
cold cost. Roomy DS4 has an 8.6% first-token penalty and overlapping whole-trace
ranges. Pressure traces finish 13.2% sooner; median forwarded tokens fall from
382,676 to 282,423 [280,282–297,774]. Policy calibration and storage completion
change which states survive, so reused work varies between repetitions.

Decode is measured from the first emitted token to the last, excluding cache
publication after the final token. Median GLM short rates are 44.231 versus
43.919 tokens/s, and DS4 pressure rates are 77.119 versus 77.675 tokens/s
(main versus PR). This does not establish a decode speedup. Speculative round
shapes and target work can vary even when emitted answers match.

Maximum after-request RSS samples, summarized across runs, are approximately
38.0 versus 39.3 GiB for GLM short and 77.9 versus 80.2 GiB for DS4 pressure.
These are sampled process RSS values, not lifetime peaks or retained-cache
payload counts. Cache budgets exclude live model arenas and allocator overhead.

## Publication improvement within the PR

This comparison holds FlashStates behavior and useful output work constant.
All three arms use the new cache, rather than main's legacy cache.

| GLM short, five runs | Original PR `3433c7d` | Retain first read | Retain first read + parallel hashes, shipped |
| --- | ---: | ---: | ---: |
| Returning first token, ms | 1,770.8 [1,751.3–1,823.5] | 1,280.2 [1,266.0–1,293.1] | 772.2 [768.7–785.6] |
| Complete trace, s | 45.644 [45.145–45.744] | 42.279 [42.012–42.468] | 39.285 [39.035–39.531] |

The shipped change reduces this PR's returning-request latency by 56.4% and its
complete short trace by 13.9%. It does not show a pressure improvement over the
original PR: pressure trace medians are 498.257 s before and 504.333 s after,
with overlapping ranges. Total request CPU time is also similar. Parallel
hashing reduces elapsed publication time, rather than eliminating all hash work.

## Recipes and correctness gates

The completed study contains **65 trials, 1,110 requests and 206,595 generated
tokens**, with no request, accounting, cache-I/O or applicable parity failures.
Main selects its legacy cache with `MAYA_STATE_CACHE=0`, legacy keep enabled,
and explicit RAM/disk/count budgets. The PR selects the shared manager with
legacy settings still supplied; its adapter disables the legacy manager.
Each arm uses a fresh private cache directory. No cache flush, clock change,
forced prefill strategy or disabled CPU path is used.

| Case | Workload | Retained budgets |
| --- | --- | --- |
| GLM short | Two agents, three turns, 4,096 initial tokens, 64 appended tokens, 128 output tokens per turn | 2 GiB RAM, no SSD |
| GLM long | Same two agents, outputs 128 / 4,096 / 1 tokens, 64 appended tokens | 2 GiB RAM, no SSD |
| DS4 roomy | Four agents, three turns, 8,192 initial tokens including 4,096 common tokens, 256 appended tokens, 64 outputs | 2 GiB RAM, 2 GiB SSD; main's 128 MiB prefix tier counts within RAM |
| DS4 pressure | Sixteen agents, three turns, 4,184–11,048 initial tokens, variable additions and 8–128 outputs | 128 MiB RAM, 128 MiB SSD, no separate prefix tier |

The pressure shape follows the FlashStates AgentX evaluation's motivation:
interleaved growing agent conversations. Counts come from
`semianalysisai/cc-traces-weka-062126` revision
`23f152f6f0f9399a85901b89a6458def0ef16729`, scaled down by eight with outputs
capped at 8–128 tokens. Natural tokenizer-corpus text supplies payloads, and
each engine's actual answer enters its next prompt. This is a scaled workload
study, not a reproduction of the paper's hardware, original content, timing or
concurrency. The pinned [FlashStates materials](FLASHSTATES-PLAN.md#reference-versions)
describe the inspiration.

All requests use temperature 0, seed 17, speculation enabled and identical EOS
biases to preserve requested output lengths. GLM uses a private copy of matched
native CPU calibration and verifies all three seed hits and startup plans;
normal runtime retiming remains enabled. DS4 uses matched private lane
calibration. The controller freezes executable, source, recipe, model-config
and reference hashes and rechecks them before each arm.

Non-long cases require identical prompt hashes and complete output-token
arrays against the original PR baseline. GLM long uses a frozen set of complete
answers independently produced by two unchanged original-PR runs. Membership
requires the exact agent, turn and prompt hash. A novel prompt or answer stops
the run. No token tolerance, count-only check or candidate-generated golden is
used. Valid long-answer variants can change later prompt text, so whole-trace
long results are closed-loop comparisons, not identical-prompt replays. The
single-prompt result above has five identical-prompt samples in both arms.

A separate instrumented C3 GLM run independently reread all 2,443,066,688 captured
bytes and read back all 813,969,792 restored GPU bytes with zero differences.
The source and behavioral tests also passed strict compilation, ASan/UBSan,
TSan and the scalar-digest configuration. Current CMake checks pass for the
shared manager, GPU state I/O and GLM/GDN checkpoint exporters. Other hardware,
models and global generation determinism are not established by this study.

## Experiments excluded from the PR

A later candidate parallelized large duplicate captures. It saved about 0.9 s
inside publication but finished the diagnostic trace about 3.1 s slower. It is
not shipped.

Another candidate reused hashes only after exact comparison with immutable RAM
bytes. Its 192 MiB duplicate microbenchmark fell from 322 ms to 27 ms, but its
45-trial macro study stopped after 38 validated trials: one DS4 pressure answer
differed by one token. That request used no restored state. Full pressure byte
checks passed, and isolated replays plus two unchanged full-trace baseline
controls did not reproduce the differing answer. The cause remains unresolved.
That candidate is not shipped, its failed matrix stays failed, and its timings
are not included in the tables above. The quality gate was not relaxed.

## Reproduction and raw evidence

The immutable completed experiment is retained on `server60` at
`/mnt/models-2tb/maya-flashstates-bench/20261113-performance-calibrated`.
It contains `trial-plan.json`, `frozen-fingerprints.json`, `run_trials.py`,
`eval_cache.py`, the exact workload JSON, calibration copies, original-baseline
references, all per-run results/logs and `aggregate-runs.json`. The adjacent
`20261113-performance` directory retains the original failed study and private
measured binaries, including `diagnostics/worker-caps/16/{maya,maya-ds4}`.

These are host-local frozen research artifacts, not an installed public
benchmark command. To replay one arm on that host without overwriting results,
use a fresh label, the frozen driver and an explicit executable. For example:

```sh
cd /mnt/models-2tb/maya-flashstates-bench/20261113-performance-calibrated
/tmp/maya-flashstates-build/venv/bin/python eval_cache.py \
  --engine flash --model glm --binary /tmp/maya-flashstates-build/cuda-make/maya \
  --label reproduction-glm-short --sessions 2 --turns 3 --common 0 \
  --base 4096 --append 64 --output 128 --ram-mb 2048 --disk-mb 0 \
  --ban-eos 154820:-100,154827:-100,154829:-100
```

For main, select `--engine main` and its pinned executable. The complete arm
arguments, source identities and reference rules are in the frozen plan. Do not
resume or overwrite the completed study. Retain new results separately and
check whole answers, prompts, work counts and final drain before comparing times.

The append-only investigation trail, code reviews, byte evidence, rejected
experiments and promotion provenance are under
`/mnt/models-2tb/project-maya/.pstack/flashstates-performance/`. The original
worktree source referenced by the first failed study was archived before
promotion; its historical fingerprint manifest was not rewritten.
