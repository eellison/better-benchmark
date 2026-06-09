# var_mean_var_mean_98a1510ddc30

## Compile: 25.38us, Oracle: 23.36us, Gap: 1.086x

## Diagnosis: EMBEDDING_PRODUCER_INLINE

## Root cause: The oracle computes the complete MBart token-embedding + position-embedding + two dependent hidden-size-1024 population-variance LayerNorms in one Triton row kernel, directly loading from embedding tables within the kernel. Inductor separates this into 2 kernels: (1) a pointwise kernel for iota + position embedding lookup, and (2) a persistent reduction kernel for word embedding lookup + add + dual LayerNorm. The 8.6% gap comes from the extra kernel launch overhead of the separate position embedding kernel.

## Kernel count
- Inductor: 2 kernels (triton_poi_fused_add_embedding_iota_unsqueeze_0, triton_per_fused_add_mul_rsqrt_sub_var_mean_1)
- Oracle: 1 kernel (embedding + dual layernorm in one pass)

## Config exploration results
| Config | Compile (us) | Ratio |
|--------|-------------|-------|
| combo + CDT + multi_kernel=1 | 29.26 | 1.253x |
| combo + CDT + multi_kernel=2 | 29.68 | 1.271x |
| combo + CDT + multi_kernel=3 | 29.16 | 1.248x |

Note: The oracle harness bench result (25.38us) uses CUDAGraph capture which reduces launch overhead. The multi_kernel exploration without CUDAGraph shows ~29us. No multi_kernel config closes the gap.

## Classification: EMBEDDING_PRODUCER_INLINE

The position embedding computation (iota + add + embedding lookup on a [1026, 1024] table) is lowered as a separate pointwise kernel because it involves indirect indexing (gather from embedding table). The oracle inlines this directly in the main row kernel, computing `position_embedding_ptr + (seq_index + offset) * hidden + cols` as a direct load. Inductor's scheduler cannot fuse a pointwise-with-gather producer into a consuming reduction kernel.

## Fix path
This pattern requires the scheduler to recognize that a small embedding lookup (position embedding with deterministic indices from iota) can be inlined into a consuming reduction kernel as a direct load. The position indices are entirely data-independent (iota + constant offset), so the gather is equivalent to a strided load.

File: `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion decisions for embedding-into-reduction)
File: `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (load generation for inlined embedding)

Related patterns: Other transformer models with position embeddings (BART, MBart, M2M100) that fuse embedding + layernorm.

## Status: design_doc

## Details
- Model: hf_MBartForCausalLM_infer_000
- Pattern: embedding(word) + embedding(position via iota+offset) -> add -> var_mean -> affine -> var_mean -> affine -> 3x aliased view outputs
- Shape: [8, 1024] input_ids, [50265, 1024] word embedding, [1026, 1024] position embedding, hidden=1024
- Inductor's main kernel is already persistent (R0_BLOCK=1024, xnumel=8192) and fuses both LayerNorms in one pass
- The gap is ONLY from the separate position embedding kernel launch (~2us overhead)
- The word embedding lookup IS fused into the main kernel (via indirect indexing from input_ids)
- Only the position embedding (which has deterministic/constant indices) is separated
- Output: 3 aliased [8192, 1024] fp32 views of the same buffer
