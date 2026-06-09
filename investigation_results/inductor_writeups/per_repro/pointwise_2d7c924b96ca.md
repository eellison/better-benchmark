# pointwise_2d7c924b96ca

## Classification: QKV_SPLIT_MATERIALIZATION_FUSION

## Current Result

- Oracle path: `repros/canonical/pointwise_2d7c924b96ca/oracle_swin_qkv_materialize.py`
- Correctness: PASS
- Oracle: 29.63 us
- Compile (cd=True): 33.66 us
- Ratio: 1.136
- Status: GOOD (gap)

## Root Cause

The oracle computes the complete Swin QKV layout materialization in one tiled Triton transpose/copy kernel. From a single `[6272, 3072]` input, it produces three outputs:
- Q: `[4096, 49, 32]` (scaled by 0.1768)
- K: `[4096, 32, 49]` (transposed)
- V: `[4096, 49, 32]` (direct copy)

All three are derived from the same input via reshape + permute + unbind, with Q getting a pointwise scale and K getting an additional transpose.

Inductor lowers the decomposed reshape/permute/unbind/scale/expand/clone/reshape graph as separate generic layout-copy regions. It cannot fuse sibling clone/materialization outputs from one QKV producer when:
1. One branch includes a pointwise scale (Q)
2. Another branch needs a tiled transpose (K)
3. A third branch is a direct copy (V)

## Kernel Count

- Inductor: 2-3 kernels (separate materialization per branch)
- Oracle: 1 kernel (tiled multi-output with per-branch epilogue)

## Config Exploration

| Config | Result |
|--------|--------|
| Default (cd=True) | ratio=1.136 |
| multi_kernel=2 | N/A (pointwise pattern, not reduction) |
| multi_kernel=3 | N/A (pointwise pattern, not reduction) |

## Diagnosis

The scheduler does not fuse sibling clone/materialization outputs from one QKV producer when branches have different epilogues (scale, transpose, identity). Each branch becomes a separate kernel reading the same input buffer.

## Design Doc

**What's needed**: Teach layout-copy scheduling to group QKV split materializations and sink simple per-branch pointwise epilogues into one multi-output tiled copy. The scheduler should recognize:
- Multiple outputs derived from the same input via view/unbind
- Each output is a layout copy (clone to contiguous) with optional simple pointwise (mul scalar) or transpose

These should fuse into a single tiled kernel that reads the input once and writes all outputs with per-branch index arithmetic.

**File references**:
- `torch/_inductor/scheduler.py` (can_fuse, score_fusion for multi-output layout copies)
- `torch/_inductor/codegen/triton.py` (multi-output tiled kernel emission)

**Affected repro count**: This Swin QKV pattern is common in vision transformers. Similar pattern in pointwise_40d7159db31f (visformer).
