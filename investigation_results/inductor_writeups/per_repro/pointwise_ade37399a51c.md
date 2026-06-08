# pointwise_ade37399a51c

## Current Result

- Family: `batch_index_gather`
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/pointwise_ade37399a51c/oracle_batch_index_gather.py`
- Correctness: PASS
- Oracle: `10.11 us`
- `torch.compile coordinate_descent_tuning=True`: `10.30 us`
- Ratio: 1.019 (AT_FLOOR)

## Config Exploration

| Config | Oracle (us) | Compile (us) | Ratio | Status |
|--------|------------|-------------|-------|--------|
| default (cd=True, combo=True) | 10.11 | 10.30 | 1.019 | AT_FLOOR |
| multi_kernel=2 | 10.05 | 10.14 | 1.01 | AT_FLOOR |
| multi_kernel=3 | 11.07 | 10.78 | 0.974 | AT_FLOOR |
| use_fast_math=True | 10.88 | 10.37 | 0.953 | AT_FLOOR |

## Diagnosis

The oracle computes a batch-axis advanced-index gather as a tiled Triton kernel that loads one selected batch index per output tile and writes fresh contiguous `[32,3,224,224]` output directly. Inductor lowers the `view -> select -> aten.index.Tensor` graph through generic advanced-index pointwise code and compiles to a single kernel. Both implementations are bandwidth-bound at the mandatory input-read plus output-write memory-traffic floor (~19.3 MB read + 19.3 MB write). The measured ratio of 1.019x is within noise.

- Inductor kernel count: 1
- Oracle kernel count: 1

## Commands

```bash
python repros/canonical/pointwise_ade37399a51c/oracle_batch_index_gather.py --check
python repros/canonical/pointwise_ade37399a51c/oracle_batch_index_gather.py --bench
```
