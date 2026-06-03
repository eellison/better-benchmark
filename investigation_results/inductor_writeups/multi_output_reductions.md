# Inductor Writeup: Multi-Output Reductions

## Status

- Queue id: `multi_output_reductions`
- Priority: P0
- Implementation status: writeup complete; patch not started
- Oracle coverage:
  - `repros/canonical/sum_sum_sum_70d71fcb0d68/oracle_multi_output_reduction.py`
  - `repros/canonical/sum_sum_sum_7b24a3457260/oracle_multi_output_reduction.py`

## Target Repros

- `sum_sum_sum_70d71fcb0d68`
- `sum_sum_sum_7b24a3457260`
- `sum_sum_cdaed89f373c`
- `sum_sum_6a14a9c9ba88`

## Objective

Reduce rereads and bad partitioning for same-input reductions and related epilogues. Optimize for best coordinate-descent/autotuned runtime.

## Oracle Gap Rationale

The oracle scaffolds isolate repeated reductions over shared source tensors. Current lowering/scheduling often emits separate reductions or big fused kernels with incompatible axes/register pressure.

## Stash Relevance

- scalar accumulators: immediate likely win
- tiling scores: candidate selection and split/fuse decisions
- `R0_BLOCK` expansion: large reductions
- persistent threshold: allow better persistent candidates

## Likely Inductor Hooks

- IR support for multi-output reductions and Welford/reduction arity.
- Scheduler cost model for same-base/same-axis sibling reductions.
- Codegen scalar accumulator handling.
- Combo partitioning when reductions have incompatible axes.

## Implementation Plan

1. Port scalar accumulator fixes from `stash@{0}` first.
2. Add cost-model preference for fusing compatible same-input reductions.
3. Add split penalties for incompatible-axis fusion.
4. Consider BN-backward semantic template after generic multi-output wins.
5. Validate on oracle-backed reps and final 3-config sweep winners.

## Validation Commands

```bash
python scripts/bench_compare.py repros/canonical/sum_sum_sum_70d71fcb0d68/repro.py \
  --config "coordinate_descent_tuning=True" --label baseline_cd \
  --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2" --label combo_persistent_cd \
  --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_looped_cd \
  --output /tmp/multi_output_compare.json

python repros/canonical/sum_sum_sum_70d71fcb0d68/oracle_multi_output_reduction.py --check --no-append
```

## Success Metric

Best tuned runtime improves on current 3-config timing, with reduced rereads and no correctness regressions.
