# pointwise_b33a9caccca9

## Current Result

- Family: `direct_padded_gather`
- Classification: `NEW_PATTERN`
- Oracle path: `repros/canonical/pointwise_b33a9caccca9/oracle_direct_padded_gather.py`
- Correctness: PASS
- Oracle: `6.91 us`
- `torch.compile coordinate_descent_tuning=True`: `7.04 us`
- Ratio: 1.019 (AT_FLOOR)

## Config Exploration

| Config | Oracle (us) | Compile (us) | Ratio | Status |
|--------|------------|-------------|-------|--------|
| default (cd=True, combo=True) | 6.91 | 7.04 | 1.019 | AT_FLOOR |
| multi_kernel=2 | 6.98 | 7.17 | 1.028 | AT_FLOOR |
| multi_kernel=3 | 6.98 | 7.17 | 1.028 | AT_FLOOR |
| use_fast_math=True | 7.01 | 7.17 | 1.023 | AT_FLOOR |

## Diagnosis

The oracle writes the indexed relative-position table directly into the padded backing layout and returns the final expanded view, avoiding intermediate materialization. The repro is from `timm_beit_base_patch16_224` and computes `index -> view -> permute -> clone -> unsqueeze -> constant_pad_nd -> slice -> expand` as one fused gather-into-padded-destination kernel. Despite the oracle's architectural advantage (direct destination-strided gather), Inductor's generic pointwise lowering compiles to a single kernel that performs nearly identically (~1-3% gap, within noise). The gap is effectively at floor because the operation is small (the `[197,197]` index table with 12 channels and 3 pad columns is only ~3.6 MB total output).

- Inductor kernel count: 1
- Oracle kernel count: 1
- The initial 1.07 measurement was noise; repeated measurements show 1.019-1.028x.

## Commands

```bash
python repros/canonical/pointwise_b33a9caccca9/oracle_direct_padded_gather.py --check
python repros/canonical/pointwise_b33a9caccca9/oracle_direct_padded_gather.py --bench
```
