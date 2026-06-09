# any_amax_sum_93fd43445eb0

## Compile: 11.74us, Oracle: 11.3us, Gap: 1.04x

## Diagnosis: BANDWIDTH_BOUND

## Root cause

The oracle computes a BERT-large attention softmax with an all-inf-row guard as a single fused persistent row-softmax Triton kernel. Inductor already decomposes the view/iota/ge/expand/where/add/eq/any/amax/sub/exp/sum/div/where/expand/view graph into one well-fused persistent reduction kernel that achieves near-identical performance.

The gap is within measurement noise (1.04x with 500 reps). The oracle's hand-written row-softmax kernel and Inductor's auto-generated persistent reduction emit essentially equivalent memory traffic patterns. The computation is bandwidth-bound on the [16, 512, 512] input with K_LEN=512 fitting entirely in registers.

## Config exploration

| Config | Compile (us) | Notes |
|--------|-------------|-------|
| default (combo_kernels=True, cdt=True) | 11.74 | Best |
| multi_kernel=2 | 35.71 | Much worse (persistent reduction forced on wrong dim) |
| multi_kernel=3 | 35.70 | Much worse |

## Kernel count
- Inductor: 1 persistent reduction kernel
- Oracle: 1 persistent row-softmax kernel

## Status: AT_FLOOR

No fix needed. The gap is within noise (1.04x). Inductor's auto-generated code is essentially optimal for this pattern.

## File references
- Oracle: repros/canonical/any_amax_sum_93fd43445eb0/oracle_full_attention_softmax.py
- Model: hf_BertForMaskedLM inference
- Pattern: view + iota mask + stable softmax + inf-row guard
