# var_mean_deb7c9191e39

## Classification: STENCIL_PRODUCER_INLINE (cheap non-broadcast producer)

## Current Result

- Family: `swin_window_reverse_shift_layernorm`
- Oracle path: `repros/canonical/var_mean_deb7c9191e39/oracle_swin_window_reverse_shift_layernorm.py`
- Correctness: PASS (max_diff=2.86e-06)
- Oracle: `113.5 us`
- `torch.compile coordinate_descent_tuning=True` (AFTER FIX): `124.5 us`
- Ratio (AFTER FIX): 1.097
- Previous ratio (BEFORE FIX): 1.411
- Status: `mostly_closed`

## Fix implemented

**Commit**: `f58d2545cd2` on `pr-184905` branch in `/tmp/pytorch-work`

Extended `inline_recomputable_producers` pass to handle "cheap non-broadcast producers".
The pass previously required at least one small read (broadcast dominance). The Swin
window-reverse + cyclic shift + residual add producer has only full-size reads (both
addmm and residual are 51M elements), but is trivially cheap (11 ops: index math + add).

Added an alternative path in `_is_inlinable_producer_shape`: if `opcount.num_ops <= 16`
and the intermediate buffer is >= 4MB, allow inlining. The existing profitability check
(`_is_inline_profitable`) still gates the decision.

Result: 2 kernels -> 1 kernel, 412MB intermediate eliminated, 1.41x -> 1.10x.

## Diagnosis

The producer `op0` (Pointwise, 51M elements, 11 ops) computes window-reverse + cyclic
shift + residual add. It reads from two full-size inputs (addmm_5 [401408, 128] and
view_24 [128, 56, 56, 128]) and outputs a [128, 56, 56, 128] intermediate.

The consumers are:
- `op1` (WelfordReduction): first pass of var_mean (mean computation)
- `op2` (WelfordReduction): second pass of var_mean (variance computation)
- `op4` (Pointwise): LN epilogue (sub, rsqrt, mul, add with weight/bias)

After inlining, all three consumers recompute the cheap index math inline instead of
reading from the 412MB materialized buffer.

## Kernel count

- Oracle: 1 kernel
- Inductor BEFORE fix: 2 kernels (poi + red)
- Inductor AFTER fix: 1 kernel (red with inlined producer)

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (lines 5681-5696: cheap producer path in `_is_inlinable_producer_shape`)
