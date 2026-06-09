# pointwise_2cb944a69993 — oracle_rope_qk

## Status: GOOD (1.052x regression — borderline)

## Benchmark Results
- Oracle: 10.46 us
- Compiled: 11.01 us
- Ratio: 1.052x

## Classification: SCHEDULER_FUSION (borderline)

## Root Cause

The oracle fuses the full Moondream Q/K RoPE application (frequency table construction, both rotary applications for Q and K, both output cats, and a constant-folded ne_scalar side output) into a single kernel with flat row-based indexing.

Inductor uses a combo_kernel with 3 sub-kernels (dispatched via `pid % 3`):
1. **Sub-kernel 0**: Computes the RoPE rotation for the first 32 dims of both Q and K (the rotary-embedding application), writing to `out_ptr0` (Q rotated) and `out_ptr1` (K rotated)
2. **Sub-kernel 1**: Copies the tail 32 dims of Q (non-rotated passthrough) from the transposed layout to the output
3. **Sub-kernel 2**: Copies the tail 32 dims of K (non-rotated passthrough) from the transposed layout to the output

The ne_scalar output is correctly constant-folded (`_frozen_param0`).

### Why Inductor Is Slightly Slower

1. **Combo kernel overhead**: The 3-way `pid % 3` dispatch means only 1/3 of thread blocks work on each sub-task. The tail-copy sub-kernels 1 and 2 are pure memory-copy work that could be folded into sub-kernel 0.

2. **Oracle's advantage**: The oracle processes each (head, position) pair as a row, loading all 64 dims at once and writing both Q and K outputs. It avoids the split between rotated (first 32) and non-rotated (last 32) portions.

3. **Magnitude**: The gap is only 5.2%, which is at the borderline of actionable. The combo_kernel approach is not unreasonable here.

## Kernel Count
- Inductor: 1 combo kernel (3 sub-kernels via pid%3)
- Oracle: 1 kernel (flat row-based iteration)

## Config Exploration Results
- `combo_kernels = True`: This is what enables the 3-way combo kernel. Without it, Inductor would emit 3 separate kernels (worse).
- `coordinate_descent_tuning = True`: Already enabled
- The combo_kernel approach is the best Inductor can do today for this pattern

## Design Doc: Potential Enhancement

### Problem
When the output is a cat of [rotated_first_half, passthrough_second_half] for both Q and K, Inductor splits this into:
- A compute kernel for the rotated half
- Copy kernels for the passthrough halves

Ideally, a single kernel would handle both halves of the cat for both Q and K, loading each input element once and writing it to the correct position in the output.

### Proposed Enhancement
Teach the scheduler to recognize that cat(compute(slice(x, 0:32)), slice(x, 32:64)) can be expressed as a single kernel iterating over the full output dimension, where the first 32 elements undergo rotation and the last 32 are passthrough copies. This avoids the combo_kernel split.

### Priority: LOW
At only 1.052x, this is barely above the threshold. The fix would primarily help RoPE patterns in models with partial rotary dimensions (like Moondream, which rotates 32 of 64 dims).

### Files to Modify
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: Recognize slice+compute+cat patterns where the cat reassembles slices of the same tensor
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: Emit unified iteration over the full cat output dimension
