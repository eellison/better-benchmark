# sum_sum_sum_fefa336f7504 - GPT-Neo Embedding Scatter Reduce

## Classification
SCATTER_REDUCE

## Benchmark Results
- Oracle: 207.74 us
- Compile (baseline, combo_kernels+coord_descent): 303.04 us
- **Ratio: 1.459x** (oracle is 45.9% faster)
- multi_kernel=2: 370.76 us (worse)
- multi_kernel=3: 333.57 us (slightly better than baseline, still far from oracle)

## Root Cause

The repro is the GPT-Neo layernorm-backward + embedding gradient computation (from `hf_GPTNeoForCausalLM_train`). It computes:
1. Layernorm backward (mul by gamma, row sums, subtract, scale by invstd)
2. Two column reductions across [32,128,2048]: `sum_dim_int_list_2` (weighted by normalized) and `sum_dim_int_list_3` (plain)
3. Position embedding scatter-add via `index_put(accumulate=True)` with indices [1,128]
4. Token embedding scatter-add via `index_put(accumulate=True)` with indices [32,128] into [50257,2048]
5. Final add of mm[:50257] + token scatter result

### Oracle approach (3 kernels):
1. Init kernel: copies mm[:50257] to vocab_out, zeros position_out
2. Main kernel (128 programs, 1 per sequence position): loops over batch, computes layernorm backward per-row, accumulates partial hidden sums, atomically scatters to position and token destinations
3. Finalize: reduces 128 partial sums -> final [2048] column sums

### Inductor approach (7 kernels):
1. `triton_poi_fused_0`: zeros position embedding output [2048,2048]
2. `triton_poi_fused_1`: zeros vocab embedding output [50257,2048]
3. `triton_per_fused_...2`: Main reduction kernel (cooperative reduction with workspace) - computes layernorm backward, produces row-level embedding gradients, writes workspace for column sums
4. `triton_mor_finalize_sum_3`: Finalize workspace for xhat_sum column reduction
5. `triton_mor_finalize_sum_4`: Finalize workspace for plain_sum column reduction
6. `triton_per_fused_...5`: Position scatter (separate kernel doing index_put + reduction over batch dim)
7. `triton_poi_fused_add_slice_6`: vocab_out = mm[:50257] + scatter result

## Kernel Count
- **Oracle: 3 kernels**
- **Inductor: 7 kernels**

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| combo+coord_descent (baseline) | 303.04 | 7 kernels |
| multi_kernel=2 | 370.76 | worse (overhead of multi-kernel dispatch) |
| multi_kernel=3 | 333.57 | slightly worse |

No config combination closes the gap significantly.

## Why Inductor Cannot Do This Today

The fundamental issue is that Inductor's scheduler treats the rowwise layernorm-backward producer, the column reductions, and the two `index_put(accumulate=True)` scatter operations as independent scheduling nodes. The oracle achieves its speedup by:

1. **Computing each row's layernorm-backward ONCE** and immediately scattering to both destinations, avoiding re-reading the row producer
2. **Accumulating column reductions in registers** across the batch loop, avoiding a separate cooperative reduction pass
3. **Fusing init + compute + scatter** into a coherent pipeline

The scheduler cannot currently:
- Keep row-local reduction scalars live while feeding multiple indexed accumulator destinations
- Recognize that index_put(accumulate=True) can be lowered as in-kernel atomic adds within a producer loop
- Fuse sibling column reductions with scatter operations sharing the same producer

## Fix: Design Doc

**Needed enhancement**: A structured embedding-backward scatter-reduce template in `torch/_inductor/fx_passes/` that pattern-matches:
- Rowwise producer (layernorm backward) feeding both:
  - Column reductions (sum over batch/seq dims)
  - Indexed scatter-add operations (index_put with accumulate=True)

**Implementation location**: `torch/_inductor/fx_passes/post_grad.py` (pattern registration) + new template in `torch/_inductor/kernel/embedding_backward_scatter.py`

**Affected repro count**: This pattern appears in all transformer training backward passes with tied embeddings (GPT-2, GPT-Neo, OPT, etc.)

## Status
Design doc only - fix not implemented. The pattern requires a new structured template that is beyond a simple scheduler tweak.
