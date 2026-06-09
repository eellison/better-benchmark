# pointwise_c2e3486ef2f2

## Current Result

- Family: `transpose_cat_pad_view`
- Classification: `BAD_ORACLE`
- Oracle path: `repros/canonical/pointwise_c2e3486ef2f2/oracle_transpose_cat_pad_view.py`
- Correctness: PASS
- Oracle: `33.50 us`
- `torch.compile coordinate_descent_tuning=True`: `26.30 us`
- Ratio: 0.785 (BAD_ORACLE)

## Diagnosis

The oracle preserves the exact eager MobileBERT `permute -> cat -> {constant_pad_nd, permute}` contract by materializing both the padded `[512,30524]` output and a separate contiguous unpadded cat backing buffer. Inductor's compiled schedule writes only the padded buffer and returns the second output as a cheaper alias view. The compiled version is 21.5% faster than the oracle because Inductor avoids the extra cat-buffer store.

This is classified as BAD_ORACLE because torch.compile is significantly faster than the hand-written oracle. There is no performance gap to close; Inductor has found a more efficient materialization strategy.

- Inductor kernel count: 1
- Oracle kernel count: 1

## Commands

```bash
python repros/canonical/pointwise_c2e3486ef2f2/oracle_transpose_cat_pad_view.py --check
python repros/canonical/pointwise_c2e3486ef2f2/oracle_transpose_cat_pad_view.py --bench
```
