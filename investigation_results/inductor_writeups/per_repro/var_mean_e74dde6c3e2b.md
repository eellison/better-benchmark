# var_mean_e74dde6c3e2b

## Classification: EMBEDDING_GATHER_LAYERNORM_FUSION

## Current Result

- Family: `blenderbot_embedding_layernorm_aliases`
- Oracle path: `repros/canonical/var_mean_e74dde6c3e2b/oracle_blenderbot_embedding_layernorm_aliases.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete Blenderbot token embedding, generated positional embedding add, fp32 hidden-size-2560 var_mean normalization, affine scale/bias, and three returned `[4096,2560]` alias views in one Triton row kernel. Inductor currently lowers the indexed embedding/add producer and generic var_mean LayerNorm template as separate scheduled regions with an intermediate `[32,128,2560]` tensor.

## Root cause

The repro performs:
1. Token embedding gather (indexed read from embedding table)
2. Positional embedding add (generated position indices)
3. Population var_mean over hidden=2560
4. Affine LayerNorm epilogue
5. Three alias-only view outputs from the same normalized storage

Inductor's norm-template scheduling does not sink row-local embedding gathers and generated position indexing into the fixed-hidden reduction loop while preserving multi-view alias metadata. This forces the embedding+add result to be materialized before the norm can proceed.

## Kernel count

- Oracle: 1 kernel (fused embedding gather + position add + LayerNorm + affine)
- Inductor: 2+ kernels (embedding+add pointwise, then norm + affine)

## Config exploration

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning=True | Minor tile improvement |
| multi_kernel=2 | Persistent may partially help for hidden=2560 |

## Recommendation

Teach the LayerNorm scheduler to fuse gather/add producers into the row reduction epilogue and return alias-only views from the single normalized output buffer. The key is recognizing that the embedding gather is a row-local operation that can be inlined into the norm kernel's load phase.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (norm producer fusion)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (gather-in-norm kernel emission)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (alias view detection)
