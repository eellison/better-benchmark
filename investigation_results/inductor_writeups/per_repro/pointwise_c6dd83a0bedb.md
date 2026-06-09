# pointwise_c6dd83a0bedb

## Current Result

- Family: `head_flatten_layout`
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/pointwise_c6dd83a0bedb/oracle_head_flatten_layout.py`
- Correctness: PASS
- Oracle: `161.60 us`
- `torch.compile coordinate_descent_tuning=True`: `161.57 us`
- Ratio: 1.0 (AT_FLOOR)

## Diagnosis

The oracle materializes the full `permute(0,2,1,3).clone()._unsafe_view(...).view([B*S,H*D])` contract by writing the transformer head/sequence transpose directly into the final contiguous `[B,S,H*D]` backing storage and returning its `[B*S,H*D]` view. Inductor lowers the decomposed permute-clone-view graph through a generic layout-copy schedule that reaches the exact same floor. The ratio of 1.0x confirms perfect parity. The operation is dominated by the mandatory ~538 MB read plus ~538 MB write for the `[175360, 768]` f32 tensor layout copy.

- Inductor kernel count: 1
- Oracle kernel count: 1

## Commands

```bash
python repros/canonical/pointwise_c6dd83a0bedb/oracle_head_flatten_layout.py --check
python repros/canonical/pointwise_c6dd83a0bedb/oracle_head_flatten_layout.py --bench
```
