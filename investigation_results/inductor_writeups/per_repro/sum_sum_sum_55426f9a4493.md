# sum_sum_sum_55426f9a4493

## Summary

- Model: ALBERT (hf_AlbertForMaskedLM_train)
- Oracle: `oracle_albert_multi_reduction.py`
- Classification: SCHEDULER_FUSION
- Ratio: 1.086x (oracle 452.5us, compile 491.4us)

## Root Cause

The repro computes the ALBERT backward fragment with a materialized layernorm-backward tensor (f32, captured operation order), then twelve product reductions, twelve plain column reductions, and twelve matrix/tensor column reductions all targeting [4096, 4096] -> [4096], plus a non-contiguous transpose view output.

The oracle fuses all 36 reductions into one 128-row partial-reduction Triton pass plus one finalizer. Inductor emits separate partial-reduction kernels for each sibling source and separate finalizers, even though the reductions share the same [4096, 4096] -> [4096] axis.

The 8.6% gap is modest because the computation is dominated by reading the many [8, 512, 4096] and [4096, 4096] input tensors (52 inputs total), and the reduction kernels themselves are relatively cheap compared to the producer pointwise.

## Config Exploration

| Config | Compile Time (us) |
|--------|-------------------|
| baseline (CDT) | 491.4 |
| multi_kernel=2 | 491.3 |
| multi_kernel=3 | 495.6 |

multi_kernel=2/3 provide no improvement. The bottleneck is separate reduction launches, not kernel config tuning.

## Fix Assessment

**Scheduler fusion** -- Same fix class as sum_sum_sum_34c857ab7db3. Teach the reduction scheduler to form a multi-output partial-reduction group for compatible sibling reductions sharing the same reduction axis.

### Difficulty: Medium
All sibling reductions have compatible shapes and reduction axes; the main barrier is that Inductor's scheduler treats each as an independent scheduling node.

## Status: open_gap

## Affected Repros

Related to sum_sum_sum_34c857ab7db3 (same model, different shape configuration, same scheduling limitation).
