# pointwise_40d7159db31f

## Classification: QKV_SPLIT_MATERIALIZATION_FUSION

## Current Result

- Oracle path: `repros/canonical/pointwise_40d7159db31f/oracle_visformer_qkv_layout.py`
- Correctness: PASS
- Oracle: 25.34 us
- Compile (cd=True): 28.29 us
- Ratio: 1.116
- Status: GOOD (gap)

## Root Cause

Same pattern as pointwise_2d7c924b96ca (Swin QKV) but from Visformer. The oracle computes the complete Visformer QKV layout materialization in one tiled Triton kernel from a single `[128, 2304, 7, 7]` (channels-last stride) input, producing three outputs:
- Q: `[768, 49, 128]` (direct layout copy)
- K: `[768, 128, 49]` (transposed)
- V: `[768, 49, 128]` (direct layout copy)

The input is channels-last (stride `(112896, 1, 16128, 2304)`) which adds complexity to the reshape+permute+unbind pattern.

Inductor separates this into multiple kernels because it cannot fuse sibling clone materializations from one QKV producer when one branch includes a transpose.

## Kernel Count

- Inductor: 2-3 kernels (separate per-branch materialization)
- Oracle: 1 kernel (tiled multi-output)

## Config Exploration

| Config | Result |
|--------|--------|
| Default (cd=True) | ratio=1.116 |
| multi_kernel=2 | N/A (pointwise, not reduction) |
| multi_kernel=3 | N/A (pointwise, not reduction) |

## Diagnosis

Same root cause as pointwise_2d7c924b96ca: the scheduler does not group sibling layout-copy outputs that share one input reader. The fix would be the same multi-output tiled layout copy enhancement.

## Design Doc

See pointwise_2d7c924b96ca for the full design doc. This is the same pattern (QKV_SPLIT_MATERIALIZATION_FUSION) applied to Visformer's channels-last input layout.

**File references**:
- `torch/_inductor/scheduler.py` (multi-output layout copy fusion)
- `torch/_inductor/codegen/triton.py` (tiled multi-output emission)

**Affected repro count**: At least 2 repros in this batch (pointwise_2d7c924b96ca, pointwise_40d7159db31f). Likely more across the corpus.
