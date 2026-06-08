# pointwise_af2e722aed5d

## Current Result

- Family: `layout_transpose`
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/pointwise_af2e722aed5d/oracle_layout_transpose.py`
- Correctness: PASS
- Oracle: `27.33 us`
- `torch.compile coordinate_descent_tuning=True`: `26.27 us`
- Ratio: 0.961 (AT_FLOOR)

## Diagnosis

The oracle materializes the attention-head layout `view(B,H,S,D).permute(0,2,1,3).clone().view(B*S,H*D)` as a fresh contiguous tensor with simplified affine indexing. Inductor lowers the same clone/view contract to a generic layout-copy kernel that achieves the same or better performance (compiled version is actually faster than oracle). Both are at the bandwidth floor for the mandatory input read and output write of the `[4096,4096]` f32 tensor (~64 MB round-trip).

- Inductor kernel count: 1
- Oracle kernel count: 1
- Status: Inductor already wins (oracle is slower)

## Commands

```bash
python repros/canonical/pointwise_af2e722aed5d/oracle_layout_transpose.py --check
python repros/canonical/pointwise_af2e722aed5d/oracle_layout_transpose.py --bench
```
