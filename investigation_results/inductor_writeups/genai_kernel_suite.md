# genai kernel suite — triage + fixes (2026-06-11, B200)

Suite: `repros/models/genai/{CrossEntropy,Softmax,LayerNorm,RMSNorm}{Forward,Backward}/full_graph_*.py`
(standalone kernel benchmarks; no partitioner-artifact risk).

Bench method: `scripts/bench_parallel.py --full-graphs repros/models/genai --device-kind B200`
(fresh `TORCHINDUCTOR_CACHE_DIR`, GPU lock, pytorch-work @ a85d79a900a / pr-184905).
CE-backward graph 001 needs `--allow-unsafe-full-graphs` (i64 target input, randint is fine
for this graph since targets only feed one-hot eq/gather).

Raw results: `results/genai_triage_2026-06-11.json`, `results/genai_triage_ceb001_2026-06-11.json`.

## Triage table (fresh re-measure; the old 2628/2248/1619/1597us attribution numbers predate
## several landed fixes — re-measured below)

`default_us` = inductor default autotune; `cd_us` = with coordinate-descent tuning.
SOL references: `optimal_kernels/*.py` hand kernels where they exist, else algorithmic
memcopy-SOL from bytes at 6.9 TB/s.

| graph | shape | default us | CD us | best-known / SOL us | default/SOL | CD/SOL | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| SoftmaxForward/000 | 8192x262144 bf16 | 3540.9 | 1913.9 | 1915 (hand, 2-pass) | 1.85 | 1.00 | default configs bad; CD = hand kernel |
| SoftmaxBackward/000 (recompute fwd) | 8192x262144 | 3526.4 | 1401.7 | ~1245 (1 read + 1 write)¹ | 2.83 | 1.13 | default configs bad |
| SoftmaxBackward/001 (dgrad) | 8192x262144 | 2879.5 | 2458.6 | 1904 (hand, 2-pass) | 1.51 | 1.29 | gap even after CD |
| CrossEntropyForward/000 | 8192x262144 | 1969.0 | 746.4 | 689 (hand, 1-pass online) | 2.86 | 1.08 | default configs bad; CD near-hand |
| CrossEntropyBackward/000 (recompute) | 8192x262144 | 1977.4 | 747.5 | ~689¹ | 2.87 | 1.08 | default configs bad |
| CrossEntropyBackward/001 (dgrad) | 8192x262144 | 1725.3 | 1435.6 | ~1245 (1 read + 1 write)² | 1.39 | 1.15 | moderate gap after CD |
| LayerNormForward/000 | 1152000x512 | 343.8 | 343.9 | 342 (SOL; hand=inductor) | 1.01 | 1.01 | at SOL |
| LayerNormBackward/000 (recompute) | 1152000x512 | 304.1 | 268.0 | ~263³ | 1.16 | 1.02 | near SOL after CD |
| LayerNormBackward/001 (dgrad) | 1152000x512 | 675.9 | 380.6 | ~343 | 1.97 | 1.11 | default configs bad |
| RMSNormForward/000 | 1152000x512 | 344.0 | 345.8 | 342 (SOL; hand=inductor) | 1.01 | 1.01 | at SOL |
| RMSNormBackward/000 (recompute) | 1152000x512 | 304.0 | 227.0 | ~225³ | 1.35 | 1.01 | near SOL after CD |
| RMSNormBackward/001 (dgrad) | 1152000x512 | 380.8 | 372.5 | ~343 | 1.11 | 1.09 | small gap |

¹ graph 000 of the backward pairs is the forward-with-saved-stats recompute: it reduces the
  large tensor to scalars/row-stats, so minimal traffic is 1 full read (+1 full write for
  SoftmaxBackward/000 which materializes y before `.sum()` — inductor can fuse so 1 read
  + small writes; CD's 1402us ≈ 2 reads, hand floor unclear).
² sum_3 in CE dgrad is algebraically -grad[row] (sum of one-hot * grad), so 1 read of x +
  1 write of dx = 8.59 GB = 1245us floor; CD 1436us.
³ row-stat outputs only: 1 full read = 1.18 GB = 171us, plus dweight partials; measured CD
  values taken as near-floor pending kernel inspection.

## Headline findings

1. The single biggest, *systematic* problem is the default reduction config search space:
   every large-rnumel (262144) graph is 1.4x-2.9x off SOL with default autotune but lands
   at/near hand-kernel time under coordinate-descent tuning. The hand-kernel docstrings
   (optimal_kernels/*.py) independently identified the same root cause: `_reduction_configs`
   caps R0_BLOCK at 1024-2048; the winning configs use R0_BLOCK=4096-8192 with XBLOCK=1.
2. SoftmaxBackward/001 (dy*y - y*sum(dy*y)) is the only graph still >1.25x off best-known
   *after* CD — needs kernel dump + blocker analysis.
3. LN/RMS forward are at SOL (B200); the segment-aligned-split + evict_first fixes already
   landed appear to have done their job for the small-rnumel family.

## Fixes (in progress)

(to be filled in per-fix: blocker file:line, hand-A/B, commit, before/after)
