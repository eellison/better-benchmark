# Validated Performance Results

All results on B200 (148 SMs), measured with `bench_parallel.py --no-share-cache`, cold inductor cache, 5 warmup + 10 rep.

## Validated Wins (ready to ship)

| Change | Geomean Speedup | Improved | Regressed | Net Savings | Repros Tested | Sweep Files |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| tiling_scores: recover INNER hint via coalescing ratio | **1.063x** | 25 | 0 | 1.41ms | 124 (single-kernel, >50us) | `sweep_tiling_ON/OFF.json` |

## Investigated but Not Net-Positive

| Change | Result | Why |
|--------|--------|-----|
| Scalar accumulators (reduce within iteration) | Net negative at R0_BLOCK=1024 | Needs R0_BLOCK≥4096 to matter, but raising R0_BLOCK hurts online_softmax |
| Online softmax disable for large rnumel | Helps softmax (32%) but doesn't fire on captured repros (pre-decomposed) | Needs post-grad pass to replace `prepare_softmax_online` |
| Split suppression (numel≥2*num_sm) | More fusion but 336 regressions | Unsplit persistent loops too much for large rnumel |
| Cooperative reduction eligibility | Works for C≥296 but epilogue re-reads input for C<296 | Cooperative RSPLIT formula broken for moderate xnumel |
| Reduction+transpose loop reorder | Fuses but 5% slower without coord descent | Non-coalesced transposed stores; needs atomic 2D approach |
| Reduction+transpose atomic 2D | 0.102ms (1.77x) on single repro | Pattern-specific, small corpus impact (~5ms total) |

## Measurement Notes

- bench_parallel had a result mis-attribution bug (fixed in `6cb0d9e0`): PREFETCH stdout pollution shifted result stream
- byte_accounting had overcounting bugs for scatter/slice (fixed in `caf82a65`)
- Baseline comparisons must match by `total_bytes` (different sweep modes use different shape keys)
- Always clear `/tmp/torchinductor_dev` AND `~/.triton/cache` between A/B runs
