# var_mean_eebe86d9365d

## Compile: 71.36us, Oracle: 65.38us, Gap: 1.092x

## Diagnosis: ROW_BLOCKING_AMORTIZATION

## Root cause: The oracle computes the DeiT distilled-token patch LayerNorm by row-blocking eight adjacent token rows in one Triton program, amortizing affine weight/bias loads and patch-layout arithmetic across neighboring rows. Inductor generates a single fused kernel (triton_red_fused_add_cat_expand_mul_permute_rsqrt_sub_var_mean_view) that handles the full expand/cat/view/permute/add/var_mean graph, but processes one row per program instance without multi-row blocking.

The gap is modest (~9%) and comes from the oracle's register-resident affine parameter reuse across 8 adjacent rows within the same CTA, reducing L2 pressure on the weight/bias loads when processing 25344 rows of hidden=768.

## Kernel count
- Inductor: 1 kernel (red_fused_add_cat_expand_mul_permute_rsqrt_sub_var_mean_view)
- Oracle: 1 kernel (row-blocked patch layernorm with 8 rows per program)

## Config exploration results
- multi_kernel=1 (default): 71.36us (ratio 1.092x)
- multi_kernel=2: 71.33us (ratio 1.093x) - no improvement
- multi_kernel=3: 71.26us (ratio 1.091x) - no improvement
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: ROW_BLOCKING_AMORTIZATION

The oracle's advantage is multi-row blocking within a single CTA to amortize affine parameter loads. This is a codegen optimization rather than a scheduling/fusion issue.

## Fix path
Enhancement in `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: For row-reduction kernels with many rows and small hidden dimensions where affine parameters (weight/bias) are invariant across rows, emit multi-row blocking (process N rows per CTA) to amortize parameter loads. This is a codegen-level tiling optimization.

## Status: design_doc
