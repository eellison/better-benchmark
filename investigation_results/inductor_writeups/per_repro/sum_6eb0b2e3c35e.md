# sum_6eb0b2e3c35e

## Current Result

- Family: `one_hot_exp`
- Classification: `ALGEBRAIC_ELIMINATION`
- Oracle path: `repros/canonical/sum_6eb0b2e3c35e/oracle_one_hot_exp.py`
- Correctness: PASS
- Oracle: `1235.9 us`
- `torch.compile coordinate_descent_tuning=True`: `1560.4 us`
- `combo_kernels + multi_kernel=2`: `1561.5 us` (no change)
- `combo_kernels + multi_kernel=3`: `1560.8 us` (no change)
- `coordinate_descent_tuning + use_fast_math`: `1386.4 us` (1.126x improvement)
- Ratio: 1.264x (oracle vs compile)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete DistilGPT2 cross-entropy-backward dense gradient, replacing the materialized one-hot row sum with the equivalent guarded label scalar formula. For each of 16384 rows across 50257 vocabulary columns, the Inductor kernel scans the full vocabulary via `iota == label` to produce a one-hot row, multiplies by the ignore-mask-scaled gradient, sums the row (which is either -scale or 0), and then uses that sum to scale the exp values. The oracle recognizes that `sum(one_hot * scale)` equals `-scale` when the label is valid (not -100) and 0 otherwise, eliminating the 50257-wide reduction entirely.

## Config exploration results

| Config | Time (us) | vs cd |
|--------|-----------|-------|
| cd (baseline) | 1560.4 | 1.000x |
| combo + mk=2 | 1561.5 | 0.999x |
| combo + mk=3 | 1560.8 | 1.000x |
| cd + use_fast_math | 1386.4 | 1.126x faster |
| Oracle | 1235.9 | 1.263x faster |

`use_fast_math=True` provides a meaningful 12.6% speedup (likely from faster exp via __expf), but still leaves a residual 12.2% gap vs the oracle. The remaining gap is the algebraic elimination of the one-hot reduction scan.

## Root cause

Inductor's algebraic simplification does not canonicalize one-hot masked reductions into per-row guarded label formulas. The pattern is: `iota == label` produces a one-hot mask; `where(one_hot, -1, 0) * scale` produces a row with exactly one non-zero; `sum(row)` scans 50257 elements to discover a scalar that is algebraically known from the label. The oracle computes `row_sum = -scale if label_valid else 0` directly, then emits `one_hot_scaled - exp_values * row_sum` per element without the scan.

## Kernel count
- Oracle: 1 kernel (tiled across rows and vocab columns, no reduction)
- Inductor: 1 kernel (fused but with inner 50257-wide reduction scan per row)

## Classification: ALGEBRAIC_ELIMINATION

The fix is to add a guarded one-hot reduction rewrite before reduction scheduling that recognizes `sum(where(iota==label, const, 0))` as the constant itself (guarded by label validity), and emit the dense epilogue directly without scanning the vocabulary.

## Note on use_fast_math

The `use_fast_math` config provides partial improvement (12.6%) by enabling faster exp intrinsics, which helps the exp-heavy epilogue. This is an orthogonal low-hanging-fruit improvement that does not address the fundamental algebraic redundancy.

## Details
- Model: hf_DistillGPT2_train_002
- Pattern: iota -> eq -> where -> mul -> sum -> exp -> mul -> sub -> view (cross-entropy backward)
- Shape: [16384, 50257] (batch=32, seq=512, vocab=50257)
- The one-hot reduction scans 50257 elements per row to produce a known scalar
