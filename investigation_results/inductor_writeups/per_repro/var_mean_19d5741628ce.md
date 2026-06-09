# var_mean_19d5741628ce

## Classification: REDUCTION_EPILOGUE_REREAD

## Current Result

- Oracle path: `repros/canonical/var_mean_19d5741628ce/oracle_mobilevit_patch_layernorm.py`
- Correctness: PASS
- Oracle: `17.47 us`
- `torch.compile coordinate_descent_tuning=True`: `20.06 us` (20.1us in bench_compare)
- `torch.compile combo_kernels=True,multi_kernel=2`: `20.5 us`
- `torch.compile combo_kernels=True,multi_kernel=3`: `21.9 us`
- Best compile config: `coordinate_descent_tuning=True` baseline at `20.06 us`
- Ratio (best compile vs oracle): 1.148x
- Status: `gap_persists`

## Diagnosis

The oracle computes the MobileViT patch-layout rewrite and fp32 hidden-dim-240 LayerNorm in one row kernel. It maps output rows directly to NCHW patch spatial indices [128,240,8,8] -> [512,16,240], performs var_mean(correction=0, keepdim=True), affine, and emits the [8192,240] view plus [512,16,1] rsqrt side output.

Inductor:
1. Cannot fuse the complex view/permute/contiguous layout semantics directly into the row kernel
2. Materializes intermediate layout transformations before the LayerNorm reduction

multi_kernel=2/3 both HARM performance (20.5us, 21.9us vs 20.1us baseline). The looped variants are worse because hidden=240 is small enough for persistent reduction.

## Kernel count
- Oracle: 1 kernel (layout rewrite + layernorm + side output fused)
- Inductor: 2+ kernels (layout materialization + norm template)

## Config exploration results
- `coordinate_descent_tuning=True`: 20.1 us (best Inductor)
- `multi_kernel=2`: 20.5 us (2% worse)
- `multi_kernel=3`: 21.9 us (9% worse -- HARMS this pattern)
- multi_kernel HARMS this pattern because it disrupts the natural persistent reduction for hidden=240

## Fix path: NEW_PATTERN -- the scheduler needs to recognize that view/permute/contiguous chains feeding LayerNorm can be expressed as indirect indexing in the row kernel prologue, avoiding materialization. For MobileViT specifically, each row is a fixed spatial 2x2 patch lane over channels that maps to simple arithmetic index computation.

## Status: design_doc
- File references: /tmp/pytorch-work/torch/_inductor/scheduler.py (view/permute fusion), /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm template)
