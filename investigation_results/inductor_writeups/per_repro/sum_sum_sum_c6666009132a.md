# sum_sum_sum_c6666009132a

## Summary

- Model: NFNet (timm_nfnet_l0_train)
- Oracle: `oracle_nfnet_silu_multi_output.py`
- Classification: ALGEBRAIC_ELIMINATION
- Ratio: 1.31x (oracle 201.7us, compile 264.0us)
- Kernel count: Oracle 2 kernels (spatial summaries + channel finalize), Inductor multiple separate reductions

## Root Cause

The oracle computes the complete `sum_sum_sum_c6666009132a` NFNet backward fragment by streaming the shared scaled-input, gate sigmoid, explicit natural-exp SiLU-derivative producer once into per-`(N,C)` spatial summaries for both returned channel reductions, whereas Inductor materializes `add_tensor_3`, materializes the spatial `[128,512,1,1]` sum, then launches separate channel reductions for the sigmoid-gradient output and sibling `add_tensor_3` sum.

Inductor cannot do this today because its algebraic simplifier/reduction codegen does not flatten the linear `sum([2,3]) -> sigmoid-derivative multiply -> sum([0,2,3])` chain into the same multi-output reduction as the sibling channel sum while preserving the captured exp/reciprocal formulation. The fix is ALGEBRAIC_ELIMINATION: teach Inductor to reassociate such dependent reductions and emit one multi-accumulator channel-reduction schedule over the shared fused producer.

## Config Exploration

| Config | Time (us) | Speedup vs default |
|--------|-----------|-------------------|
| baseline (default) | 292.7 | 1.00x |
| coordinate_descent_tuning=True | 264.1 | 1.11x |
| combo_kernels=True | 292.7 | 1.00x |
| triton.multi_kernel=2 | FAIL (ND reduction assertion) | N/A |
| triton.multi_kernel=3 | FAIL (ND reduction assertion) | N/A |

CDT provides a modest 11% improvement but does not close the 31% gap to the oracle. multi_kernel modes fail on this repro due to an ND reduction shape assertion. The gap is structural -- requires algebraic reassociation of dependent reductions.

## Fix Assessment

**Classification: ALGEBRAIC_ELIMINATION** -- requires teaching Inductor to recognize when `sum(spatial_dims) -> pointwise -> sum(batch_dims)` can be collapsed into a single multi-accumulator reduction pass that produces both the intermediate spatial result (after sigmoid-derivative application) and the final channel sum simultaneously.

### What's needed:
A reduction-reassociation pass that detects when a chain of reductions over different dimensions can be linearized into a single pass with multiple accumulators, especially when the intermediate reduction is followed by a cheap pointwise (sigmoid gradient) before the outer reduction.

### Difficulty: Medium-High
The pattern is well-defined (dependent reductions with linear pointwise between them) but the analysis to prove legality and emit correct multi-accumulator code requires careful handling of the algebraic properties.

## Commands

```bash
python repros/canonical/sum_sum_sum_c6666009132a/oracle_nfnet_silu_multi_output.py --check
python repros/canonical/sum_sum_sum_c6666009132a/oracle_nfnet_silu_multi_output.py --bench --warmup 25 --rep 200
python scripts/bench_compare.py repros/canonical/sum_sum_sum_c6666009132a/repro.py --config "baseline" --label "default" --config "coordinate_descent_tuning=True" --label "cdt" --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True" --label "combo" --n-warmup 10 --n-rep 50 --rounds 5
python -m py_compile repros/canonical/sum_sum_sum_c6666009132a/oracle_nfnet_silu_multi_output.py
python scripts/validate_corpus_invariants.py
```

## Affected Models

- timm_nfnet_l0_train (4 variants in corpus)
