# sum_sum_sum_34c857ab7db3

## Summary

- Model: ALBERT (hf_AlbertForMaskedLM_train)
- Oracle: `oracle_albert_gelu_multi_sum.py`
- Classification: BANDWIDTH_BOUND
- Ratio: 1.127x (oracle 619.4us, compile 698.3us)

## Root Cause

The repro computes the ALBERT tanh-GELU backward fragment: a materialized GELU-backward tensor (non-contiguous [16384, 4096] transpose view), plus eleven sibling [4096, 16384] -> [16384] column reductions, and a final sibling add with the GELU column sum.

The oracle uses a two-kernel pointwise-then-multi-reduction envelope with natural `libdevice.tanh` formulation that closely mirrors what Inductor selects. The 12.7% gap comes from Inductor launching separate reduction kernels for each of the 11 sibling column sums rather than grouping them into one multi-accumulator pass.

However, the gap is modest because both paths are dominated by the large memory traffic: 13 input tensors of shape [4096, 16384] f32 (each ~256MB) must be read regardless.

## Config Exploration

| Config | Compile Time (us) |
|--------|-------------------|
| baseline (CDT) | 698.3 |
| multi_kernel=2 | 701.2 |
| multi_kernel=3 | 700.7 |

multi_kernel=2/3 provide no improvement. The gap is structural (separate reduction launches) not config-tunable.

## Fix Assessment

**Scheduler fusion** -- The fix would teach Inductor's reduction scheduler to group multiple same-axis column reductions ([4096, 16384] -> [16384]) that share compatible input lifetimes into a single multi-accumulator partial-reduction kernel, eliminating redundant input re-reads.

### Difficulty: Medium
The sibling reductions all have the same reduction axis and compatible shapes, but the scheduler currently treats each as an independent node.

## Status: open_gap

## Affected Repros

Similar ALBERT column-reduction patterns likely affect sum_sum_sum_55426f9a4493 (same model, different shape).
