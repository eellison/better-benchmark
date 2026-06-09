# pointwise_9827abedee25

## Classification: MULTI_OUTPUT_LAYOUT_MATERIALIZATION

## Current Result

- Oracle path: `repros/canonical/pointwise_9827abedee25/oracle_qkv_layout.py`
- Correctness: PASS
- Oracle: 28.35 us
- Compile (cd=True): 34.14 us
- Ratio: 1.204
- Status: GOOD

## Root Cause

The oracle computes the complete Swin QKV split scope in one Triton pass: the [B, S, 3, H, D] view/unbind, Q scaling, three required clone materializations, and exact returned non-contiguous view layouts with distinct fresh storages. It writes Q/K/V backing tensors by reading the source once.

Inductor lowers the metadata-heavy layout graph (view, unbind, permute, clone, scale) through generic pointwise/layout-copy scheduling, resulting in separate kernel launches for each output materialization.

## Kernel Count

- Oracle: 1 kernel (single pass writing Q, K, V)
- Inductor: Multiple kernels (separate layout copies)

## Config Exploration

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.204 |
| multi_kernel=2 | 1.083 |
| multi_kernel=3 | 1.083 |

multi_kernel=2/3 reduces the gap from 1.204 to 1.083 but does not fully close it.

## File/Line References

- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: separate scheduling of sibling layout materializations
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: each clone/permute generates a separate kernel

## Design Doc

Similar to pointwise_78060a0d4a6b (Visformer QKV). The fix requires multi-output layout-copy scheduling that reads one shared source and writes multiple outputs with different index maps in a single kernel. This is bandwidth-bound work dominated by memory movement; the remaining ~8% gap after multi_kernel is largely launch overhead from separate kernels.

Best config: multi_kernel=2 or 3 (reduces to 1.083).
