# pointwise_78060a0d4a6b

## Classification: MULTI_OUTPUT_LAYOUT_MATERIALIZATION

## Current Result

- Oracle path: `repros/canonical/pointwise_78060a0d4a6b/oracle_visformer_qkv_train_layout.py`
- Correctness: PASS
- Oracle: 24.38 us
- Compile (cd=True): 28.22 us
- Ratio: 1.157
- Status: GOOD

## Root Cause

The oracle computes the complete Visformer training Q/K/V layout split in one Triton multi-output layout kernel. It writes three fresh base storages by reading the QKV source once and scattering to Q [768,128,49], K [768,49,128] (transposed), and V [768,128,49] outputs simultaneously.

Inductor lowers the sibling unbind/permute/clone/reshape/permute materializations as separate generic layout-copy work because its scheduler does not fuse multiple users of the same strided QKV producer when each output has a different final index map.

## Kernel Count

- Oracle: 1 kernel (multi-output layout scatter)
- Inductor: Multiple kernels (separate layout copies for Q, K, V)

## Config Exploration

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.157 |
| multi_kernel=2 | 1.089 |
| multi_kernel=3 | 1.086 |

multi_kernel=3 slightly helps (reduces ratio from 1.157 to 1.086) but does not fully close the gap.

## File/Line References

- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: cannot fuse sibling layout-copy consumers with different index maps into one multi-output kernel
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: each layout copy is a separate pointwise kernel

## Design Doc

The fix requires teaching layout-copy scheduling to emit a multi-output fused producer for sibling Q/K/V materializations with per-output affine store maps. When N consumers all read the same source but write to different output layouts, Inductor should be able to emit one kernel that reads the source once and writes all N outputs with distinct store index expressions.
