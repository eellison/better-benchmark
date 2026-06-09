# amax_sum_f5fb3a355fc1

## Classification: ALGEBRAIC_ELIMINATION

## Current Result

- Family: `online_softmax`
- Oracle path: `repros/canonical/amax_sum_f5fb3a355fc1/oracle_online_softmax.py`
- Model: hf_GPTNeoForCausalLM_train_000, hf_GPTNeoForSequenceClassification_train_000
- Correctness: PASS (max_diff=1.19e-07)
- Oracle: 15.78 us
- Compile (default harness, coordinate_descent_tuning=True): 20.22 us
- Ratio: 1.282
- Status: GOOD (gap exists)

## Config exploration results

| Config | Median (us) | Notes |
|--------|-------------|-------|
| default (harness, CUDAGraph) | 20.22 | ratio 1.282 |
| multi_kernel=2 (no CUDAGraph) | 52.4 | launch overhead dominates |
| multi_kernel=3 (no CUDAGraph) | 52.7 | launch overhead dominates |

The harness measurements use CUDAGraph capture which eliminates launch overhead. The standalone multi_kernel tests without CUDAGraph show ~52us due to kernel launch costs dominating at this small problem size.

## Root cause

The oracle computes the GPT-Neo scalar-fill masked additive-bias softmax in a single Triton kernel that:
1. Views [512,128,128] to [32,16,128,128]
2. Slices a [1,1,2048,2048] bool mask to [1,1,128,128]
3. Applies scalar fill for masked-off positions
4. Adds broadcast [32,1,128,128] bias
5. Computes stable last-dimension softmax (amax/sub/exp/sum/div)
6. Writes output with final contiguous [512,128,128] view layout

The kernel tiles across heads (block_h=HEADS=16), processing one (batch, q) position across all heads in each CTA. This exploits the fact that the sliced mask is shared across all heads (shape [1,1,128,128]) and the bias is broadcast across heads (shape [32,1,128,128]).

Inductor lowers this as a single fused kernel with online-softmax but:
1. Does not exploit the head-shared mask (reloads for each head independently)
2. Cannot tile across heads because the mask-slice + where pattern breaks the standard tiling
3. At K_LEN=128 (small reduction dim), the overhead of separate reductions is proportionally larger

The 28% gap at this small shape (512*128=65K elements per row, 512*16 = 8K rows) comes from:
- Mask/bias reload overhead (mask is [128,128] = 16KB, shared across 16 heads)
- Kernel launch efficiency (small problem, fewer CTAs)

## Kernel count
- Oracle: 1 kernel (head-tiled, reuses mask across heads)
- Inductor: 1 kernel (online softmax, does not exploit mask sharing)

## Fix path

ALGEBRAIC_ELIMINATION / NEW_PATTERN: The fix requires recognizing that a sliced broadcast mask (shape [1,1,Q,K]) is invariant across the head dimension and can be loaded once per (batch, q) position rather than per (batch, head, q). This is a scheduling/codegen optimization:
1. Detect slice of a broadcast tensor that is invariant over the reduction's outer loop dimension
2. Hoist the mask load outside the head iteration
3. Tile across heads to amortize the mask/bias load

Relevant files:
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (persistent softmax tiling)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (broadcast-invariant detection)

## Status: design_todo

This is a NEW_PATTERN that requires recognizing broadcast-invariant sliced masks in attention softmax. The fix would benefit all GPT-Neo-family models and similar architectures using masked attention with broadcast masks.
