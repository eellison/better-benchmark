# sum_sum_sum_ffedd26b0c42

## Classification: SCATTER_REDUCE

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_ffedd26b0c42/oracle_structured_select_scatter_reduce.py`
- Correctness: PASS
- Oracle: `10.14 us`
- `torch.compile coordinate_descent_tuning=True`: `12.35 us` (13.0us in bench_compare)
- `torch.compile combo_kernels=True,multi_kernel=2`: `13.4 us`
- `torch.compile combo_kernels=True,multi_kernel=3`: `13.3 us`
- Best compile config: `coordinate_descent_tuning=True` baseline at `12.35 us`
- Ratio (best compile vs oracle): 1.218x
- Status: `gap_persists`

## Diagnosis

The oracle computes the DeiT select-scatter layer-norm-backward by treating the zero-filled `select_scatter` as a structured single-token producer. It derives the token-0 row reduction directly from `mm`, writes the transposed side output, and accumulates all three returned channel reductions from the same fused pass. Shape: [128,197,192] with select_scatter at token index 0.

Inductor materializes the dense [128,197,192] zero/select_scatter tensor and schedules the row reductions, sibling channel reductions, and permute side output as separate generic work.

multi_kernel=2/3 HARMS performance here (13.3-13.4us vs 12.35-13.0us baseline), likely because the looped/persistent reduction variants don't help a scatter-dominated pattern with small hidden=192.

## Kernel count
- Oracle: 1 kernel (structured scatter with fused reductions)
- Inductor: multiple kernels (select_scatter materialization + separate reductions)

## Config exploration results
- `coordinate_descent_tuning=True`: 12.35-13.0 us (best Inductor)
- `multi_kernel=2`: 13.4 us (worse)
- `multi_kernel=3`: 13.3 us (worse)
- multi_kernel HARMS this pattern by breaking the natural kernel structure

## Fix path: SCATTER_REDUCE -- add a structured select-scatter lowering that maps sparse token sources directly into row-reduction epilogues, emits required materialized scatter stores, and accumulates compatible channel reductions without materializing the dense scatter input.

## Status: design_doc
- File references: /tmp/pytorch-work/torch/_inductor/lowering.py (select_scatter impl), /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion decisions)
