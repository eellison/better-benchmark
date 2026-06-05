# sum_sum_a55455ad36bb

## Compile: 421.79us, Oracle: 444.13us, Gap: 0.95x (BAD_ORACLE)

## Diagnosis: BANDWIDTH_BOUND

## Root cause

The oracle computes the Inception-style masked batch-norm-backward fragment by sharing the sibling `sum(where)` and `sum(where * centered)` channel reductions, then reuses the finalized per-channel scalars to emit both the returned f32[128,32,149,149] gradient tensor and f32[32] side vector. However, Inductor's compiled output is actually 5% faster than the hand-written oracle.

The oracle is slower likely due to suboptimal tiling for the large spatial dimensions (149x149 = 22201 elements per channel per batch). With N=128, the total reduction domain per channel is 128*22201 = 2,841,728 elements, which is very large. Inductor's auto-tuned persistent/looped reduction strategy handles this better than the oracle's fixed cooperative split approach.

## Config exploration

| Config | Compile (us) | Notes |
|--------|-------------|-------|
| default (combo_kernels=True, cdt=True) | 421.79 | Already faster than oracle |

## Kernel count
- Inductor: 2 kernels (fused reduction + pointwise)
- Oracle: 3 kernels (partial reduce + finalize + epilogue)

## Status: BAD_ORACLE (compiler wins)

The oracle is 5% slower than Inductor's compiled output. No gap to close.

## File references
- Oracle: repros/canonical/sum_sum_a55455ad36bb/oracle_masked_bn_backward_full_scope.py
- Model: Inception-style BN backward
- Pattern: Masked channel reductions with dependent tensor/vector epilogue (N=128, C=32, H=W=149)
