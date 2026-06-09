# var_mean_1b02c576e1eb

## Classification: REDUCTION_EPILOGUE_REREAD

## Current Result

- Oracle path: `repros/canonical/var_mean_1b02c576e1eb/oracle_embedding_position_layernorm.py`
- Correctness: PASS
- Oracle: `18.21 us`
- `torch.compile coordinate_descent_tuning=True`: `19.2 us` (19.2us in bench_compare)
- `torch.compile combo_kernels=True,multi_kernel=2`: `19.3 us`
- `torch.compile combo_kernels=True,multi_kernel=3`: `18.0 us`
- Best compile config: `multi_kernel=3` at `18.0 us`
- Ratio (best compile vs oracle): 0.988x (gap CLOSED, Inductor FASTER)
- Status: `closed_by_config`

## Diagnosis

The oracle computes the GPT-Neo token embedding gather + position embedding gather + deterministic boolean mask + fp32 hidden-2048 LayerNorm in one row kernel. With `multi_kernel=3`, Inductor achieves 18.0us -- actually faster than the oracle's 18.21us.

The default config (19.2us) has a 5.4% gap because it uses a suboptimal reduction variant for hidden=2048. multi_kernel=3 forces the looped reduction which is better for this shape.

## Kernel count
- Oracle: 1 kernel (embedding gather + layernorm + mask fused)
- Inductor (multi_kernel=3): 1-2 kernels but optimized reduction

## Config exploration results
- `coordinate_descent_tuning=True`: 19.2 us (baseline, 1.054x gap)
- `multi_kernel=2`: 19.3 us (neutral/slightly worse)
- `multi_kernel=3`: 18.0 us (1.07x faster than baseline, BEATS oracle)

## Fix path: Gap fully closed by `triton.multi_kernel=3`. No additional fix needed.

## Status: closed_by_config
- File references: /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy)
