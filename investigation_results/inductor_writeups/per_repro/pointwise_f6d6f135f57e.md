# pointwise_f6d6f135f57e

## Classification: LAYOUT_INDEXING_STENCIL_FUSION

## Current Result

- Family: `layout_fused_swiglu`
- Oracle path: `repros/canonical/pointwise_f6d6f135f57e/oracle_layout_fused_swiglu.py`
- Correctness: PASS
- Oracle: `24.19 us`
- `torch.compile coordinate_descent_tuning=True`: `50.78 us`
- Ratio: 2.099
- Best config: `combo+mk=2`: `49.32 us` (minimal improvement)
- Status: `real_gap`

## Diagnosis

The oracle fuses the split-indexed SwiGLU-style pointwise math with the final cat+permute layout store into one full-scope Triton kernel. The output shape is [1536, 16384] bf16.

Inductor must schedule the arithmetic through the captured split/cat/permute view pattern and cannot reliably sink this producer into the transposed output layout. The scheduler treats the layout reconstruction as a separate indexing concern, resulting in an extra materialization of the intermediate buffer.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 50.78 |
| combo+mk=2 | 49.32 |
| combo+mk=3 | 49.42 |
| Oracle | 24.19 |

No config closes the 2.1x gap.

## Root cause

The scheduler/codegen does not fuse split/cat/permute layout indexing into the producing pointwise store. The oracle computes the SwiGLU math and writes directly to the transposed output layout in one kernel, avoiding the intermediate buffer round-trip.

## Kernel count
- Oracle: 1 kernel (fused SwiGLU + layout store)
- Inductor: 2+ kernels (pointwise SwiGLU, then layout copy/permute)

## Recommendation

Fix in `torch/_inductor/scheduler.py` and `torch/_inductor/codegen/triton.py`: teach scheduler to recognize split/cat/permute as indexing that can be sunk into the producing pointwise kernel's store. This is a significant gap (2.1x) affecting SwiGLU/GLU patterns with non-trivial output layouts.
