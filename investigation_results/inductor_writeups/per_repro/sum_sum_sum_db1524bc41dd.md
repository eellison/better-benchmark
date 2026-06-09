# sum_sum_sum_db1524bc41dd


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 1529.76 us
- Ratio: N/A

## Queue Position

- Rank: 44
- Family: `multi_output_reduction_templates`
- Owner: `Curie`
- Closure status: `fix_identified_split_k_cooperative`
- Oracle status: `oracle_implemented_needs_measurement`

## Current Gap

- Best compile: `1529.760 us`
- Memcopy SOL: `547.936 us`
- Launch-adjusted SOL gap: `2.717x`
- Oracle path: `repros/canonical/sum_sum_sum_db1524bc41dd/oracle_multi_output_reduction.py`
- Measured oracle: _pending GPU measurement_
- Shape label: PyTorch UNet batch-norm backward

## Gap Diagnosis

Classification: **MULTI_OUTPUT_REDUCTION** (fused dual-reduce + pointwise-reduce)

Three reductions over dims `[0,2,3]` of a `[8, 64, 640, 959]` tensor:
1. `sum1[c]` = channel sum of `where_self` (ReLU backward mask)
2. `sum2[c]` = channel sum of `where_self * sub_tensor` (BN backward mean correction)
3. `sum3[c]` = channel sum of the pointwise output `(where_self - sub*s2_scaled - s1_scaled) * weight`

The third reduction depends on the first two, creating a two-phase dependency. Inductor currently generates 5 separate kernels: it cannot fuse the third reduction with the pointwise because it materializes the full `[8, 64, 640, 959]` intermediate.

**Root cause**: Inductor does not recognize that the pointwise output feeding the third reduction is never stored -- it is consumed only by a reduction. This is a "reduction of a pointwise of reductions" pattern where the intermediate can be eliminated.

**Fix**: Fuse into two phases in a single cooperative/split-K launch:
- Phase 1: Dual-reduction kernel computing sum1 and sum2 with one read of the inputs.
- Phase 2: Fused pointwise + reduction computing sum3 without materializing the `[8, 64, 640, 959]` intermediate (reads inputs again but avoids the write + read of the large intermediate).

## Oracle Implementation

- **Triton Phase 1**: `_dual_reduce_kernel` -- iterates over spatial positions in blocks of 128, accumulates sum1 and sum2 per-channel with atomic_add across programs.
- **Triton Phase 2**: `_pointwise_and_reduce_kernel` -- reads the same inputs again, loads per-channel sum1/sum2/rsqrt/weight constants, computes the BN-backward pointwise element-wise, and accumulates sum3 without writing the intermediate.
- **Key insight**: By not materializing the `[8, 64, 640, 959]` pointwise result (3.77 GB total_bytes across all kernels), the oracle trades one extra read pass for eliminating one large write + read pair. Net memory traffic is reduced significantly for this memory-bound workload.
- **Total bytes**: 3,770,942,720 (5 kernels in compiled version)
- **Shape**: N=8, C=64, H=640, W=959; total_spatial=4,910,080; scale_factor=2.037e-07

## Inductor Closure Path

- Implementation track: Multi-output reduction fusion with dependent-reduction elimination.
- Candidate hook: Detect `[reduce1, reduce2] -> pointwise -> reduce3` pattern where the pointwise output has no other consumers; fuse phase 2 into a read-compute-reduce pass that skips the intermediate store.
- Benchmark policy: compare default (5 kernels), `coordinate_descent_tuning=True`, split-K cooperative variant, and oracle floor.
- Gating policy: gate on BN-backward pattern (dual reduction feeding pointwise feeding third reduction, intermediate has no other consumer) and shape predicate (N*H*W > threshold).

## Done Criteria

- Oracle correctness verified on GPU (--check passes with rtol=1e-3, atol=1e-3).
- Oracle timing measured and appended to measured_oracle_floors.csv.
- Inductor path either reaches the oracle floor or has a gated implementation plan.
