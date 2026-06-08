# pointwise_ba25116c7e2e

## Current Result

- Family: `segment_causal_mask`
- Classification: `BAD_ORACLE`
- Oracle path: `repros/canonical/pointwise_ba25116c7e2e/oracle_segment_causal_mask.py`
- Correctness: PASS
- Oracle: `15.04 us`
- `torch.compile coordinate_descent_tuning=True`: `12.96 us`
- Ratio: 0.862 (BAD_ORACLE)

## Diagnosis

The oracle materializes the GPT-2 segment-aware causal attention-mask bias by writing only the required fp32 `[32,1,512,512]` backing buffer and returning the exact `[32,12,512,512]` zero-stride head expand view. However, Inductor's compiled version significantly outperforms the oracle (12.96 us vs 15.04 us). This indicates the oracle's specialized segment-causal-mask pattern is not faster than Inductor's generic pointwise handling for this shape.

This is classified as BAD_ORACLE because torch.compile is 14% faster than the oracle. There is no performance gap to close.

- Inductor kernel count: 1
- Oracle kernel count: 1

## Commands

```bash
python repros/canonical/pointwise_ba25116c7e2e/oracle_segment_causal_mask.py --check
python repros/canonical/pointwise_ba25116c7e2e/oracle_segment_causal_mask.py --bench
```
