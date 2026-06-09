# pointwise_ccfd75b90652

## Current Result

- Family: `direct_pad_tail`
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/pointwise_ccfd75b90652/oracle_direct_pad_tail.py`
- Correctness: PASS
- Oracle: `50.94 us`
- `torch.compile coordinate_descent_tuning=True`: `50.91 us`
- Ratio: 0.999 (AT_FLOOR)

## Diagnosis

The oracle computes the complete double-permute plus three-row `constant_pad_nd` scope by stripping the canceling metadata-only permutes and using a linear Triton copy for the input prefix plus a zero fill for the final pad rows. Inductor lowers the same scope as a generic pointwise pad materialization after view canonicalization. The ratio of 0.999x confirms perfect parity. The output contract requires reading every input element of the `[50268, 768]` (or equivalent) f32 tensor and writing a fresh contiguous padded output, putting both at the bandwidth floor (~148 MB round-trip).

- Inductor kernel count: 1
- Oracle kernel count: 1

## Commands

```bash
python repros/canonical/pointwise_ccfd75b90652/oracle_direct_pad_tail.py --check
python repros/canonical/pointwise_ccfd75b90652/oracle_direct_pad_tail.py --bench
```
