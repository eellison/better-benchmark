# pointwise_c91baf62713c

## Current Result

- Family: `scaled_head_transpose`
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/pointwise_c91baf62713c/oracle_scaled_head_transpose.py`
- Correctness: PASS
- Oracle: `27.14 us`
- `torch.compile coordinate_descent_tuning=True`: `26.43 us`
- Ratio: 0.974 (AT_FLOOR)

## Diagnosis

The oracle directly fills the fresh contiguous clone storage for the scaled `view(B,S,H,D).permute(0,2,1,3)` layout and returns the same final permuted view. Inductor already lowers the captured view/permute/mul/expand/clone/view/permute scope to one fused layout materialization kernel that is actually slightly faster than the oracle. The operation is the `[512, 64, 512]` f32 scaled transpose (~64 MB read + 64 MB write), firmly at bandwidth floor.

- Inductor kernel count: 1
- Oracle kernel count: 1
- Status: Inductor matches or beats oracle.

## Commands

```bash
python repros/canonical/pointwise_c91baf62713c/oracle_scaled_head_transpose.py --check
python repros/canonical/pointwise_c91baf62713c/oracle_scaled_head_transpose.py --bench
```
