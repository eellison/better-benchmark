# var_mean_a88cd01397d1

## Current Result

- Family: `var_mean`
- Classification: `NEW_PATTERN`
- Oracle path: `repros/canonical/var_mean_a88cd01397d1/oracle_gpt2_dropout_layernorm.py`
- Correctness: PASS
- Oracle: `38.82 us`
- `torch.compile`: `43.04 us`
- Ratio: 1.109x
- Parent status: `not_fixable_config`

## Diagnosis

The oracle computes the full GPT-2 token+position embedding, stochastic dropout,
hidden-size-768 LayerNorm (var_mean over dim=[2]), affine epilogue, and output
transpose in a single row-parallel Triton kernel (one thread block per row of
8192 rows). It also produces the adjacent-position bool mask (ne_scalar) as a
side output within the same kernel.

Inductor generates **2 kernels**:
1. Main fused kernel: embedding lookups + dropout + var_mean + normalize + affine
   + transpose (this already does an excellent job fusing the main compute path)
2. Pointwise kernel: the `ne_scalar` bool mask computation (iota, expand, slice,
   sub, cat, ne operations)

### Root Cause

The gap (1.109x = ~4.2us) comes primarily from the second kernel launch overhead
for the bool mask computation. The bool mask logic (computing whether adjacent
position indices differ) is a cheap integer operation but requires a separate
kernel because Inductor's scheduler does not merge unrelated pointwise computations
into the main reduction kernel when they share no data dependencies with the
reduction path.

The oracle computes the ne_scalar mask within the main kernel (just stores False
for each row since positions are sequential), while Inductor must launch a separate
kernel to compute it from the position indices.

### Inductor Kernel Count

2 kernels

### Configs Tried

- Default: 2 kernels, 43.04us
- The main reduction kernel is already well-fused; the gap is purely the extra launch

### Fix Assessment

Not fixable via config. The improvement is modest (1.109x). The issue is that
Inductor cannot merge the independent bool mask computation into the main
reduction kernel since they operate on different data. A pattern-level optimization
could recognize that the ne_scalar computation on sequential position indices
always produces False, but this would be extremely narrow.

## Commands

```bash
python repros/canonical/var_mean_a88cd01397d1/oracle_gpt2_dropout_layernorm.py --check
python repros/canonical/var_mean_a88cd01397d1/oracle_gpt2_dropout_layernorm.py --bench
python -m py_compile repros/canonical/var_mean_a88cd01397d1/oracle_gpt2_dropout_layernorm.py
python scripts/validate_corpus_invariants.py
```
