# Inductor Writeup: Softmax Backward / Attention Backward

## Status

- Queue id: `softmax_backward_attention`
- Priority: P0
- Implementation status: writeup complete; patch not started
- Oracle coverage: `repros/canonical/sum_sum_sum_afd118ca907d/oracle_softmax_backward.py`

## Target Repro

- `sum_sum_sum_afd118ca907d`: T5-style attention backward fragments with `p = exp(scores - m) / l`, gradient dot-products, and `p * (g - sum(g*p))` style reductions.

## Objective

Optimize for best runtime with coordinate descent and forced looped/non-looped combo configs. Compile time and broader benchmarking are acceptable.

## Oracle Gap Rationale

Launch-adjusted priority rank is very high. The oracle scaffold splits out the core softmax-backward/attention-backward math and documents a fused Triton target. Current generic lowering recomputes/materializes fragments and does not recognize the saved `m/l` softmax-backward structure.

## Stash Relevance

- online softmax fixes may provide pattern-search/codegen base
- scalar accumulators help reduction outputs
- tiling scores may help row/block candidate selection
- `R0_BLOCK` expansion may help large attention rows

## Likely Inductor Hooks

- FX/post-grad pattern matching for `exp(x - m) / l` followed by `sum(g*p)` and `p*(g-sum)`.
- Lowering to a fused softmax-backward template when `m` and `l` are saved inputs.
- Scheduler/codegen support for fused row reductions plus output epilogues.
- Combo partitioning to avoid mixing incompatible index_put outputs with row-softmax fragments.

## Implementation Plan

1. Inspect generated graph/code for `sum_sum_sum_afd118ca907d` under final coordinate-descent configs.
2. Identify repeated softmax-backward subgraphs from oracle scaffold.
3. Add a pattern or template candidate for `p = exp(x-m)/l; out = p*(g-sum(g*p))`.
4. Keep index_put/bucket outputs separate initially; optimize core row fragments first.
5. Compare against oracle scaffold and tuned combo variants.

## Validation Commands

```bash
python scripts/bench_compare.py repros/canonical/sum_sum_sum_afd118ca907d/repro.py \
  --config "coordinate_descent_tuning=True" --label baseline_cd \
  --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2" --label combo_persistent_cd \
  --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_looped_cd \
  --output /tmp/softmax_backward_compare.json

python repros/canonical/sum_sum_sum_afd118ca907d/oracle_softmax_backward.py --check --no-append
```

## Success Metric

Best tuned runtime should reduce the gap to oracle; initial target is improving the core softmax-backward fragments without regressing correctness of all repro outputs.
