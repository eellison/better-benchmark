# var_mean_65e90900fd65

## Classification: PERSISTENT_THRESHOLD

## Current Result (after fix)

- Family: `bn_training_relu`
- Oracle path: `repros/canonical/var_mean_65e90900fd65/oracle_bn_training_relu.py`
- Correctness: PASS
- Oracle: `11.58 us`
- `torch.compile coordinate_descent_tuning=True` (with fix): `11.33 us`
- Ratio: **0.96x** (AT_FLOOR - compile matches oracle)
- Previous ratio: 1.32x
- Status: `fixed`

## Fix

Commit `417c00958be` in `/tmp/pytorch-work` on branch `pr-184905`:
- File: `torch/_inductor/choices.py` - `should_use_persistent_reduction()`
- Change: On Blackwell+ GPUs (cc >= 10.0), raise the persistent reduction threshold
  for INNER reductions from 4096 to 32768, enabling persistent mode for BN-training
  patterns where rnumel = N*H*W (typically 3136-8192).

## Diagnosis

The repro is batch normalization training with shape [128, 384, 8, 8]:
- Reduction over [N, H, W] = [128, 8, 8] = 8192 elements per channel
- 384 channels (programs)
- Epilogue: affine (weight*x + bias) + ReLU + running stats updates

**Before fix**: Inductor generated a single fused kernel using **looped reduction**
with 2 passes over the data (first pass: compute mean/var via Welford, second pass:
apply normalization). Despite the kernel being fully fused (1 kernel, not 2+), reading
data twice causes a ~32% penalty vs the oracle which reads once.

**After fix**: Inductor generates a **persistent reduction** kernel that reads
all 8192 elements per channel into registers once, computes mean/var, and
immediately applies the normalization epilogue - matching the oracle's approach.

## Root cause (revised from original diagnosis)

The original diagnosis was incorrect in stating Inductor uses 2+ kernels. Inductor
actually produces a SINGLE fully-fused kernel. The real issue is the **reduction
strategy within that kernel**: looped (2-pass, reads data twice) vs persistent
(1-pass, reads data once).

The persistent reduction threshold (`persistent_reduction_threshold_inner = 4096`)
was too low to cover BN-training patterns (rnumel = 8192). On Blackwell with
65536 regs/SM, persistent with rnumel=8192 is safe: XBLOCK=1, num_warps=8
(256 threads), 32 elements/thread = 32 registers for data, 384 programs on
148 SMs = full occupancy.

## Kernel count
- Oracle: 1 kernel (per-channel BN + affine + ReLU + running stats, 1 pass)
- Inductor (before): 1 kernel, but 2-pass looped reduction (reads input 2x)
- Inductor (after): 1 kernel, single-pass persistent reduction (reads input 1x)

## Config exploration results

| Config | Time (us) | Notes |
|--------|-----------|-------|
| Default (cd=True, before fix) | 14.24 | looped, 2-pass |
| With fix (persistent) | 11.33 | single-pass, matches oracle |
| multi_kernel=3 (force persistent, before fix) | 13.72 | persistent available via MK |
| Oracle | 11.58 | |

## Files modified
- `/tmp/pytorch-work/torch/_inductor/choices.py` (persistent threshold logic)
- `/tmp/pytorch-work/torch/_inductor/runtime/triton_heuristics.py` (occupancy num_warps)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (mutation_renames cycle fix)
