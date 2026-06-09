# var_mean_e74dde6c3e2b

## Classification: EMBEDDING_GATHER_LAYERNORM_FUSION

## Current Result

- Family: `blenderbot_embedding_layernorm_aliases`
- Oracle path: `repros/canonical/var_mean_e74dde6c3e2b/oracle_blenderbot_embedding_layernorm_aliases.py`
- Correctness: PASS (max_diff=2.86e-06, 3 outputs all pass)
- Oracle: `23.36 us`
- `torch.compile coordinate_descent_tuning=True`: `26.27 us`
- Ratio: 1.125
- Best config: default (cd=True) at `26.27 us`
- Status: `real_gap`

## Diagnosis

The oracle computes the complete Blenderbot token embedding gather, positional embedding add (position_id = row % seq_len), fp32 hidden-size-2560 var_mean normalization, affine scale/bias, and three returned [4096, 2560] alias views all in one Triton row kernel. It loads token embeddings via `token_id` gather and positional embeddings via `row % seq_len` indexing directly within the reduction loop.

Inductor emits 2 kernels: one for the embedding gather + positional add (materializing an intermediate [32, 128, 2560] buffer ~39 MB), and one for the var_mean LayerNorm + affine.

## Root cause

The norm-template scheduling does not sink row-local embedding gathers and generated position indexing into the fixed-hidden reduction loop while preserving multi-view alias metadata. The embedding lookup is treated as a separate producer that materializes before the norm can proceed because the gather pattern (indexed embedding table load) is not recognized as a fusible row-local producer for the reduction template.

## Kernel count

- Oracle: 1 kernel (fused embedding gather + position add + LayerNorm + affine, 3 alias outputs)
- Inductor: 2 kernels (embedding gather + add, then norm + affine)

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 26.27 | 1.125 |
| combo+mk=2 | 32.79 | 1.404 |
| combo+mk=3 | 31.71 | 1.357 |
| fast_math=True | 33.61 | 1.439 |
| Oracle | 23.36 | 1.000 |

No config closes the gap. multi_kernel and fast_math actually make performance worse. The oracle uses only standard `tl.rsqrt` -- no fast/imprecise transcendentals.

## Recommendation

Teach the LayerNorm scheduler to fuse gather/add producers into the row reduction epilogue and return alias-only views from the single normalized output buffer. The key is recognizing that the embedding gather is a row-local operation that can be inlined into the norm kernel's load phase.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (norm producer fusion)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (gather-in-norm kernel emission)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (alias view detection)
