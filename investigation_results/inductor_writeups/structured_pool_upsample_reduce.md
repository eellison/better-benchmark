# Inductor Writeup: Structured Pool/Upsample Scatter Reduce

## Status

- Queue id: `structured_pool_upsample_reduce`
- Priority: P0
- Implementation status: writeup complete; patch not started
- Oracle coverage:
  - `repros/canonical/sum_18262b26934c/oracle_maxpool_direct_reduce.py`
  - `repros/canonical/sum_sum_sum_f90d684d32cb/oracle_structured_upsample_reduce.py`

## Target Repros

- `sum_sum_sum_45f02142ecfd`
- `sum_sum_sum_f90d684d32cb`
- `sum_sum_sum_dadf6aa035dd`
- `sum_18262b26934c`
- `sum_sum_8bcd6e12dcd4`

## Objective

Remove dense scatter/index_put/scatter_add materialization when the scatter result only feeds masks and channel reductions. Tune/benchmark under coordinate descent and forced looped/non-looped configs.

## Oracle Gap Rationale

The top raw and launch-adjusted gaps are dominated by materializing huge dense scatter/upsample/pool-backward tensors, then rereading them for channel reductions. Oracle scaffolds compute the needed channel reductions directly instead.

## Stash Relevance

- scalar accumulators may help channel-reduction outputs
- tiling scores may help template selection
- combo kernel changes may avoid bad scatter+reduction fusion but do not solve dense materialization alone

## Semantic Rewrite

Match patterns like:

```text
full(0) + index_put/scatter_add(accumulate=True) -> add/scatter output -> where(mask or ReLU gate) -> sum([0,2,3])
```

when the dense scatter output has no non-reduction consumers.

Lower to structured templates:

- maxpool-offset direct channel reduce
- bilinear upsample/index_put direct channel reduce
- optional multi-output BN/ReLU backward reductions

## Likely Inductor Hooks

- FX/post-grad graph pattern pass for scatter/index_put followed by reductions.
- Lowering replacement to semantic composite op, e.g. `_inductor_structured_scatter_reduce`.
- TritonTemplate or codegen path for direct channel reductions.
- Scheduler guard to avoid comboing atomic scatter kernels with unrelated reductions.

## Implementation Plan

1. Start with `sum_18262b26934c` maxpool-offset direct reduce because it has a simpler oracle.
2. Add graph pattern detection and a disabled/experimental lowering path.
3. Move to `sum_sum_sum_f90d684d32cb` bilinear upsample after maxpool path works.
4. Validate shape/generalization across `45f`, `f90`, `dadf`, and `8bcd`.
5. Benchmark only with coordinate-descent forced configs.

## Validation Commands

```bash
python scripts/bench_compare.py repros/canonical/sum_18262b26934c/repro.py \
  --config "coordinate_descent_tuning=True" --label baseline_cd \
  --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2" --label combo_persistent_cd \
  --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_looped_cd \
  --output /tmp/structured_pool_compare.json

python repros/canonical/sum_18262b26934c/oracle_maxpool_direct_reduce.py --check --no-append
python repros/canonical/sum_sum_sum_f90d684d32cb/oracle_structured_upsample_reduce.py --check --no-append
```

## Success Metric

Eliminate dense materialization and significantly reduce tuned runtime. Initial target: multi-x speedup on `f90`/`dadf`, and large absolute savings on `45f`.
