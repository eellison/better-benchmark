# pointwise_b954be2ccd22

## Current Result

- Family: `affine_alias`
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/pointwise_b954be2ccd22/oracle_affine_alias.py`
- Correctness: PASS
- Oracle: `35.84 us`
- `torch.compile coordinate_descent_tuning=True`: `35.10 us`
- Ratio: 0.979 (AT_FLOOR)

## Diagnosis

The oracle computes the full MobileBERT affine pointwise scope (add/mul/add) in one Triton pass and returns three `[32768, 512]` outputs as metadata views aliasing one computed buffer. Inductor already lowers the same graph through its generic pointwise scheduler and preserves the single-buffer aliasing pattern. The compiled version is actually slightly faster than the oracle. The remaining work is dominated by the mandatory two activation reads, two broadcast parameter reads, one output store, and launch overhead.

- Inductor kernel count: 1
- Oracle kernel count: 1

## Commands

```bash
python repros/canonical/pointwise_b954be2ccd22/oracle_affine_alias.py --check
python repros/canonical/pointwise_b954be2ccd22/oracle_affine_alias.py --bench
```
