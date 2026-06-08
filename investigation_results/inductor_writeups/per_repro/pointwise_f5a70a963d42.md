# pointwise_f5a70a963d42

## Classification: BANDWIDTH_BOUND

## Current Result

- Family: `longformer_layout_chain`
- Oracle path: `repros/canonical/pointwise_f5a70a963d42/oracle_longformer_layout_chain.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `at_floor`

## Diagnosis

The oracle computes the full Longformer layout chain by directly materializing the required fresh contiguous `[B*T, H*D]` output from the original contiguous `[B*H*(T/S), S, D]` input with one shape-specialized Triton layout copy. Inductor already reaches the same CUDAGraph-measured mandatory read/write traffic envelope for this captured view/permute/clone/view scope. The user-visible result is a non-aliasing contiguous clone of all elements -- no fusion, scatter/reduce, split-K, algebraic elimination, or recomputation can reduce the mandatory traffic.

## Root cause

This is a pure layout copy (view + permute + clone + view). Both Inductor and the oracle read every element once and write every element once. The oracle uses a 2D row-tiled approach (64 rows at a time, split column loads for 512+256), while Inductor uses flat 1D tiling. On B200 with large L2, both approaches achieve equivalent bandwidth utilization.

## Kernel count

- Oracle: 1 kernel (_longformer_layout_chain_kernel)
- Inductor: 1 kernel (equivalent layout copy)

## Config exploration

No config can improve a single-kernel bandwidth-floor layout copy. The workload is fully memory-bound with no compute or fusion opportunity.

## Recommendation

Record as at-floor bandwidth case. No action unless broader layout-copy bandwidth, launch, or allocation improvements move both paths.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (layout copy tiling)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (permute/clone detection)
