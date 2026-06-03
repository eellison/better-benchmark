# sum_sum_sum_afd118ca907d

## Queue Position

- Rank: 3
- Family: `online_softmax_cross_entropy`
- Owner: `Confucius`
- Closure status: `measured`
- Oracle status: `BAD_ORACLE`
- Diagnosis: `SCHEDULER_FUSION`

## Current Gap (measured 2026-06-03, B200)

- Best compile (CD+scatter_reduce_fusion): `4275.1 us`
- Oracle (torch_direct_oracle, unoptimized): `28425.8 us` (BAD - 6.6x slower)
- Bandwidth floor (IO=4547MB): `784.0 us`
- Realistic floor (2-pass for two attention blocks): `1568.0 us`
- Compile / realistic floor: `2.73x`
- Oracle path: `repros/canonical/sum_sum_sum_afd118ca907d/oracle_softmax_backward.py`

## Oracle State

- torch_direct_oracle measured at 28425.8 us (torch ops, not optimized Triton).
- Compile is 6.6x faster than the scaffold oracle.
- A proper Triton oracle for fused softmax backward would be needed.

## Pattern Analysis

T5 attention backward (torchbench_hf_T5) with dual attention blocks:
1. First attention block:
   - 5 adds of [8,8,1024,1024] view tensors
   - dropout_backward * softmax_backward (mul * sum * fma pattern)
   - sum over batch dim [0] for relative position bias gradient
   - index_put for position bias accumulation
2. Second attention block (identical structure):
   - Same pattern but with causal mask (where) + exp/div (online softmax recomputation)
   - Same softmax_backward + position bias grad + index_put

Many [8,8,1024,1024] tensors (256 MB each) are read. The two attention blocks
are independent but share an output buffer (full_default for index_put).

## Inductor Closure Path

- Implementation track: SCHEDULER_FUSION.
- The 2.73x gap to realistic floor comes from:
  1. Many independent subgraphs not fused into fewer kernels
  2. The two attention blocks could share memory/scheduling
  3. The softmax backward (mul*softmax*sum - neg*sum pattern) generates separate reduction + pointwise kernels
- Opportunity: fused softmax-backward template that does the reduction + fma in one pass.
- The index_put operations (position bias grad) are small and fast.

## Done Criteria

- Oracle needs proper Triton fused-attention-backward implementation.
- Scheduler fusion to reduce kernel count is the main optimization vector.
