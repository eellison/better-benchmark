# MM repro corpus

Generated with producer-PW ownership and actual ATen directory prefixes.

Extraction and policy documentation: [matmul_partitioning.md](../docs/matmul_partitioning.md).

```bash
python scripts/extract_matmul_corpus.py --output mm_repro --workers 4
python scripts/validate_matmul_repros.py mm_repro
```

- 373 unique repros; 624 shape points; 5092 occurrences.
- 264 repros include epilogue computation; 63 are genuine prologue-only cases.
- All 373 files and 624 shape points reloaded and passed FakeTensor execution.
- Full extraction: 81.7s with four CPU workers. GPU compilation/autotuning was not run.
- Source MM / selected MM union: 17,527 / 7,311. Unexpected standalone MM+noop outputs: 0.

## Current distributions

| Metric | min | p25 | p50 | p75 | p95 | p100 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| ATen/repro | 2 | 7.0 | 11.0 | 20.0 | 34.0 | 86 | 14.7399 |
| ATen/shape point | 2 | 7.0 | 10.0 | 19.0 | 29.85 | 86 | 13.359 |
| MM/repro | 1 | 1.0 | 1.0 | 2.0 | 3.0 | 4 | 1.4129 |
| MM/shape point | 1 | 1.0 | 1.0 | 2.0 | 3.0 | 4 | 1.3942 |
| Shapes/repro | 1 | 1.0 | 1.0 | 2.0 | 4.0 | 22 | 1.6729 |

ATen counts include ATen views/casts; p100 is the maximum; quantiles use linear interpolation.

## Actual examples

- [Four-MM chain](canonical/addmm_8ed26adc744a/repro.py): 16-wide intermediates with tanh between MMs.
- [Horizontal projections into a BMM](canonical/addmm_bmm_fe4891df3e8d/repro.py): two projections, then K=32 batched contraction.
- [SwiGLU gating](canonical/mm_bdd1d6cf4c6a/repro.py): 1 repro / 2 shapes / 60 occurrences; the large down projection is a boundary.
- [MM plus GELU and layer normalization](canonical/addmm_4fb63f9ccc45/repro.py): M=32768, K=256, N=128.

Small vertical regions: 26 repros / 31 shapes / 167 occurrences. Pure chains and mixed horizontal/vertical regions both occur.

## Data

- [Interactive statistics](statistics.html)
- [Full report](report.json)
- [Inventory](inventory.json)
- [Source-to-repro node mapping](occurrences.json); per-shape MM geometry is in `canonical/*/shapes.json`.
- [Coverage](coverage.json)
- [FakeTensor validation](validation.json)

The corpus contains regions extracted from saved BB graphs. These heuristics select candidate fusion tasks; single GPU kernel compilation has not been certified.
