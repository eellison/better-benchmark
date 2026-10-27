# H100 GEMM backend benchmark

[Interactive report (Meta internal, read-only)](https://artifacts.internalmeta.com/a/2231427581590737)

The report compares ATEN, TRITON, NVGEMM, and CUTLASS for the first stored shape
of each of the 387 matmul repros. Timings cover the whole compiled repro.
Numerical validation is deferred.

Run the sweep with BB's parallel benchmark runner:

```bash
python scripts/bench_mm_backends.py \
  --gpus 0,1,2,3 \
  --cutlass-dir /path/to/cutlass \
  --output results/mm_gemm_h100
```

The output directory contains `comparison.csv`, `comparison.json`, and
`per_repro/<repro_id>.json`, including timings, failures, and the fastest
successful backend for each repro.
