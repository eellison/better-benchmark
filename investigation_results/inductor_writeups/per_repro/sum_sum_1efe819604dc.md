# sum_sum_1efe819604dc

## Current Result

- Family: `structured_pool_upsample_backward_reduce`
- Classification: `SCATTER_REDUCE`
- Oracle path: `repros/canonical/sum_sum_1efe819604dc/oracle_structured_pool_upsample_reduce.py`
- Correctness: PASS
- Oracle: `74.752 us`
- `torch.compile coordinate_descent_tuning`: `107.553 us`
- `torch.compile combo_looped_cd`: `120.098 us`
- Best compile/oracle gap: `1.44x`

## Diagnosis

The oracle computes the full MobileViT adaptive-average-pool backward, SiLU
derivative, and BN-backward tuple from the original pooled gradient and the
strided convolution activation. It folds the logical
`as_strided_scatter -> as_strided -> expand -> div` producer into direct
`mm[n, c] / 64` loads, accumulates both channel reductions in the same channel
program, and writes the required contiguous `[128, 640, 8, 8]` input gradient
plus `[640]` gamma gradient. Inductor currently treats the scatter/expand
producer and the reduction/pointwise consumers as ordinary scheduled work; the
actionable fix is a `SCATTER_REDUCE` lowering for structured average-pool
backward that can fuse the producer, sibling reductions, and dependent
BN-backward epilogue without materializing the expanded pool-gradient tensor.

## Commands

```bash
python -m py_compile repros/canonical/sum_sum_1efe819604dc/oracle_structured_pool_upsample_reduce.py
python repros/canonical/sum_sum_1efe819604dc/oracle_structured_pool_upsample_reduce.py --check
python repros/canonical/sum_sum_1efe819604dc/oracle_structured_pool_upsample_reduce.py --bench --warmup 10 --rep 50
```
