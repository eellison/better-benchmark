# sum_sum_3c5efeeb4188

## Status

- Family: `masked_bn_backward`
- Closure status: `open_gap`
- Artifact: `repros/canonical/sum_sum_3c5efeeb4188/oracle_masked_bn_backward.py`
- Classification: `SCHEDULER_FUSION`
- Label: `torchbench_phlippe_resnet_train_001`

## Full-Scope Contract

The oracle consumes the same inputs as `repro.py`:
- `f32[128, 64]` mm (linear output)
- `b8[128, 64, 8, 8]` dropout/ReLU mask
- `f32[128, 64, 8, 8]` activation
- `f32[1, 64, 1, 1]` mean
- `f32[64]` invstd, `f32[64]` weight
- Two shape parameters

Returns: `f32[128, 64, 8, 8]` BN input gradient, `f32[64]` channel reduction.

## Timings

- Oracle: 7.84 us
- torch.compile (coordinate_descent_tuning=True, combo_kernels=True): 8.99 us
- Ratio: 1.147x

## Kernel Count

- **Oracle**: 1 kernel (_masked_bn_backward_kernel) - single fused reduction+epilogue per channel
- **Inductor**: 1 kernel (triton_red_fused_div_expand_full_mul_sub_sum_unsqueeze_view_where_0)

## Root Cause

Both the oracle and Inductor use a single kernel, but the oracle achieves better
performance through its per-channel tiling strategy:

The oracle launches one threadblock per channel (64 blocks), with each block:
1. Loading all N*HW=8192 elements for that channel
2. Computing both reductions (grad_sum and centered_grad_sum) in registers
3. Using the reduction results immediately to compute the full epilogue output
4. Writing both the f32[64] scalar output AND the f32[128,64,8,8] full output

Inductor's approach likely uses a reduction that iterates the same N*HW elements
but with different tiling choices. The 1.147x gap suggests the oracle's tiling
(one channel per block, 8192-element reduction, immediate epilogue write) is
slightly more efficient than Inductor's autotuned parameters for this specific shape.

The gap is small (1.147x) and represents a tuning/tiling difference rather than
a fundamental algorithmic issue. The N*HW=8192 reduction dimension is small enough
for a single-pass per-channel kernel, and the oracle uses exactly this approach
while Inductor may split the work differently.

## Config Exploration

| Config | Time (us) | Notes |
|--------|-----------|-------|
| combo+cdt (default harness) | 8.99 | 1 kernel |

## Design Doc

**Why Inductor cannot match this today**: The gap is a tiling/autotuning issue.
Inductor emits a fused reduction kernel but its tiling choices (XBLOCK, R0_BLOCK,
num_warps) may not be optimal for this specific shape (C=64, N*HW=8192). The
oracle's per-channel strategy (1 block per channel, full N*HW reduction in one shot)
may be outside Inductor's autotuning search space.

**What enhancement is needed**: The coordinate descent tuning or reduction heuristic
in `torch/_inductor/choices.py` could benefit from:
1. Trying a per-channel configuration (XBLOCK=1, R0_BLOCK=8192) for BN-backward shapes
2. Ensuring the "fused reduction + epilogue" pattern has the full-channel-per-block
   variant in its search space

This is a minor tuning gap (1.147x) that may close with better autotuning coverage
rather than requiring structural changes.

**Affected files**:
- `torch/_inductor/choices.py` (reduction block size candidates)
- `torch/_inductor/codegen/triton.py` (tiling search space)

## Validation

- `oracle_masked_bn_backward.py --check`: PASS
- `oracle_masked_bn_backward.py --bench`: ratio 1.147x, status GOOD
