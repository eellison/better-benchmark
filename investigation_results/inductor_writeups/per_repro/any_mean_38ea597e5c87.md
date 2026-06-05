# any_mean_38ea597e5c87

## Classification: `AT_FLOOR`

## Pattern

T5 RMSNorm with 12 aliasing view outputs: the repro computes a residual add,
global `any(isinf)` check, conditional clamping, f16->f32 cast, row-wise
mean-of-squares, rsqrt (eps=1e-6), weight multiplication, and returns 12
aliasing `[2048, 512]` f16 views over one `[1, 2048, 512]` base buffer.

From model `torchbench_hf_T5_infer_000`.

## Measurements

```bash
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/any_mean_38ea597e5c87/oracle_t5_rmsnorm_aliases.py --check
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/any_mean_38ea597e5c87/oracle_t5_rmsnorm_aliases.py --bench
```

| Metric | Value |
|--------|-------|
| Oracle | 11.07-11.39 us |
| torch.compile (coord descent) | 10.69-12.0 us |
| Ratio | 0.95-1.08 (noise) |
| Status | AT_FLOOR |

Correctness: PASS (12 outputs, shape=[2048, 512] f16, max_diff=3.91e-03)

Extended measurement (warmup=50, rep=500):
- Run 1: oracle=11.39, compile=11.81, ratio=1.037
- Run 2: oracle=11.23, compile=10.69, ratio=0.952

## Config Exploration

| Config | Time (us) |
|--------|-----------|
| baseline | ~20.5 (no CUDAGraph) |
| coord_descent | ~27.2 (no CUDAGraph) |
| combo_cd | ~32.2 (no CUDAGraph) |

Note: non-CUDAGraph timings are higher due to kernel launch overhead from 3
kernels. With CUDAGraph (as used by bench_oracle harness), both oracle and
compile converge to ~11 us.

## Diagnosis

Inductor generates the same 3-kernel structure as the oracle:
1. `triton_red_fused_add_any_isinf_view_0`: partial isinf reduction (128 blocks)
2. `triton_per_fused_add_any_isinf_view_1`: global any reduction (scalar)
3. `triton_per_fused_add_clamp_max_clamp_min_convert_element_type_full_mean_mul_neg_pow_rsqrt_view_where_2`: fused clamp+rmsnorm+weight (2048 rows, R0_BLOCK=512)

The oracle uses the same 3-kernel approach (clear_flag + any_inf + rmsnorm).
With CUDAGraph capture eliminating launch overhead, both are bandwidth-bound
at the same level. The 12 aliasing view outputs are metadata-only and do not
affect kernel performance. No Inductor fix is needed.
