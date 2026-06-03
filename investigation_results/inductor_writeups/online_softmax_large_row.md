# Inductor Writeup: Online Softmax Large Row

## Status

- Queue id: `online_softmax_large_row`
- Priority: P0
- Implementation status: ready for Inductor patching
- Oracle coverage: `repros/canonical/amax_sum_sum_dc96a87ba8db/oracle_softmax_sum.py`

## Target Repros

- `amax_sum_sum_dc96a87ba8db`: clean `bf16 -> f32 -> amax -> exp -> sum -> div -> bf16 -> sum`, shape `[8192, 262144]`.
- `amax_sum_3ed297ef02cd`: large online-softmax-style reduction target.
- `amax_sum_f0661488d68c`: CE/softmax-style row target.

## Objective

Optimize for best measured runtime under coordinate descent/autotuning, not default heuristic selection. Compare:

- `coordinate_descent_tuning=True`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`

## Oracle Gap Rationale

The clean oracle avoids generic multi-pass/materialized softmax lowering and uses online row reduction. Existing oracle/prototype data in `optimal_kernels/softmax_fwd_8192x262144.py` and `online_softmax_heuristic_data.json` shows large-row online softmax should beat generic lowering for `rnumel >= 4096`, especially huge vocab rows.

## Stash Relevance

Use `stash@{0}` selectively:

- online softmax fixes
- `R0_BLOCK` expansion
- persistent threshold changes
- `num_warps=2`
- scalar reduction accumulators

The queue already notes effective stash analysis in `investigation_results/online_softmax_analysis.md`.

## Likely Inductor Hooks

- Reduction config generation / large `R0_BLOCK` search in Triton heuristics.
- Online softmax lowering/pattern choice for `amax -> sub -> exp -> sum -> div`.
- Triton codegen accumulator handling for scalar/reduced outputs.
- Autotune candidate expansion for forced coordinate-descent target configs.

## Implementation Plan

1. Review `stash@{0}` for online softmax and large-row reduction changes.
2. Port minimal safe changes for expanded `R0_BLOCK`, `num_warps=2`, and scalar accumulator correctness.
3. Ensure `[8192, 262144]` and similar large rows include online candidates under coordinate descent.
4. Do not optimize default-only heuristics; allow benchmark/autotune to choose.
5. Validate against final 3-config results and oracle scaffold.

## Validation Commands

```bash
python scripts/bench_compare.py repros/canonical/amax_sum_sum_dc96a87ba8db/repro.py \
  --config "coordinate_descent_tuning=True" --label baseline_cd \
  --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2" --label combo_persistent_cd \
  --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_looped_cd \
  --output /tmp/online_softmax_compare.json

python repros/canonical/amax_sum_sum_dc96a87ba8db/oracle_softmax_sum.py --check
```

## Success Metric

Best tuned runtime should move toward the oracle floor and beat current final `combo_looped`/`combo_persistent` timing for target repros.
