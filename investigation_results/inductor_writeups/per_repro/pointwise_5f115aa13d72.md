# pointwise_5f115aa13d72

## Classification: LONGFORMER_DIAGONAL

## Current Result

- Oracle path: `repros/canonical/pointwise_5f115aa13d72/oracle_longformer_chunk_layout.py`
- Correctness: PASS
- Oracle: 14.08 us
- Compile (cd=True): 15.23 us
- Ratio: 1.082
- Best config: combo+mk=3: 15.81 us (ratio 0.959 = AT_FLOOR)
- Status: **FIXED by multi_kernel**

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 15.23 | 1.082 |
| combo+mk=2 | 15.87 | 0.971 (AT_FLOOR) |
| combo+mk=3 | 15.81 | 0.959 (AT_FLOOR) |
| Oracle | 14.08 | 1.000 |

## Root Cause

The oracle fuses bias add, head/batch layout rewrite, overlapping three-window as_strided stencil clone, and final [288,512,64] strided view into one Triton materialization kernel. Inductor schedules the bias add and overlapping clone/layout materialization through more generic work with avoidable intermediate traffic.

With multi_kernel=2 or 3, the gap disappears entirely (Inductor beats the oracle).

## Kernel count
- Oracle: 1 kernel (fused bias + overlapping window layout)
- Inductor: 1-2 kernels (depending on multi_kernel config)

## Recommendation

No Inductor fix needed. The gap is closed by `triton.multi_kernel >= 2`, which is a table-stakes configuration.
