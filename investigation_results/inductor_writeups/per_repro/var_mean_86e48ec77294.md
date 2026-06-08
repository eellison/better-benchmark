# var_mean_86e48ec77294

## Classification: BAD_ORACLE

## Current Result

- Family: `reformer_layernorm_aliases`
- Oracle path: `repros/canonical/var_mean_86e48ec77294/oracle_reformer_layernorm_aliases.py`
- Correctness: PASS
- Oracle: `19.04 us`
- `torch.compile coordinate_descent_tuning=True`: `15.81 us`
- Ratio: 0.830 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's Reformer population LayerNorm kernel with duplicate view alias outputs is 17% slower than Inductor on this hardware. The oracle attempts to compute var_mean(correction=0), rsqrt, affine, and emit two identical [32768, 256] view aliases from one output buffer, but Inductor's generic Welford-based fused kernel outperforms it for this shape (32768 rows x 256 hidden).

At hidden=256, Inductor's reduction kernel is well-tuned and the oracle's approach of replacing Welford with direct sum-divide does not yield a benefit. The oracle overhead likely comes from its fixed tiling strategy being suboptimal for this narrow hidden dimension.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
