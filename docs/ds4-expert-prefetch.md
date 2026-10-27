# DeepSeek expert read-ahead controls

Maya normally issues read-ahead hints for every supplied expert range on Linux. On Windows it first checks eight working-set samples and skips ranges that look resident. These defaults are unchanged.

Linux users can opt into sampled filtering and choose the number of probes per expert range in their saved config:

```json
{
  "env": {
    "DS4_EXPERT_PREFETCH_MODE": "sampled",
    "DS4_EXPERT_PREFETCH_SAMPLES": "4"
  }
}
```

Restart Maya after changing these settings. They govern best-effort expert read-ahead hints, not weight reads, precision, model memory allocation or Engram prompt prefetch.

| Setting | Default | Behavior |
| --- | --- | --- |
| `DS4_EXPERT_PREFETCH_MODE` | `always` on Linux; `sampled` on Windows | `always` hints every supplied range. `sampled` skips a range only when all sampled pages look resident. |
| `DS4_EXPERT_PREFETCH_SAMPLES` | 4 on Linux; 8 on Windows | Decimal integer from 1 through 16, active only in `sampled` mode. Setting a count alone does not enable sampling on Linux. |

The engine validates both settings before loading weights and logs the effective policy, count and configuration source once. Values are case-sensitive. Empty values, unknown modes, signed or fractional counts, whitespace and out-of-range counts are errors, even when sampling is inactive. Settings are cached for the engine process; no calibration changes the defaults.

## Why this is opt-in on Linux

The original platform split in [#275](https://github.com/mw00/project-maya/pull/275) was a cost decision: Linux read-ahead advice for cached ranges measured cheaper than checking residency on the tested Core i5-12600T. Windows page walks were more expensive, so filtering paid there. [#425](https://github.com/mw00/project-maya/pull/425) extended filtering to the IQ CPU-expert path on Windows and deliberately left Linux unchanged; it did not benchmark Linux.

A later Linux experiment on a Threadripper PRO 3945WX, 123 GiB RAM and four RTX 3090s found a benefit with DeepSeek-V4.1-Flash GSQ-RCO IQ2_XXS and DSpark. Three fresh-engine runs per arm compared always-hint against four-sample filtering, with 12 CPU threads, mmap experts on, effective TP1, 4K prefill chunks and 262144 context. Median decode improved from 41.01 to 45.57 tok/s for a 128-token prompt/256-token answer and from 39.46 to 42.77 tok/s for a 50K-token prompt/512-token answer. The 50K request's total time improved only 2.26%, from 45.59 to 44.56 seconds, because prefill was unchanged. Matched generated-token hashes and speculative-draft counts agreed.

Those are measurements of the earlier four-sample implementation, not a benchmark of this PR against current main. Benefit has been demonstrated only on that system and those workloads. Four samples are a starting point, not a measured optimum. Other Linux systems, models, sample counts and cache conditions can be slower. Compare matched requests on your own system before keeping the setting.

## Compatibility and limits

- `DS4_EXPERT_PREFETCH_MODE` takes precedence over the existing `DS4_PF_ALL` override. Without an explicit mode, a nonzero integer in `DS4_PF_ALL` keeps always-hint behavior. Zero or an unrecognized legacy value keeps the platform default: always on Linux, sampled on Windows.
- `DS4_IQ_PF_ALL` remains a separate IQ call-site override. A nonzero integer bypasses the shared policy for those IQ ranges, even with explicit `sampled` mode. The IQ path logs that override once when used. Clear this override when comparing the shared modes.
- One sample checks only the start of a range. Any finite count can miss a cold page, and residency can change after a probe. Failed queries retain the hint; an unsampled cold page still loads its exact bytes on demand, but the request can take longer.
- Full CPU-expert RAM copies can bypass this file-backed work. A policy log does not prove that filtering is on a request's critical path.

CPU-only Linux tests cover defaults and overrides, configurable counts, invalid settings, cached parsing, failed residency queries, warm/cold ranges and demand reads after a missed cold page. The file-cache test evicts only its own temporary fixture, never model files or the host's global page cache. Native Windows execution and current-branch model performance remain unverified.
