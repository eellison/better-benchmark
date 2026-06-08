# pointwise_bcd0a32063dd

## Current Result

- Family: `embedding_affine_aliases`
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/pointwise_bcd0a32063dd/oracle_embedding_affine_aliases.py`
- Correctness: PASS
- Oracle: `28.48 us`
- `torch.compile coordinate_descent_tuning=True`: `28.51 us`
- Ratio: 1.001 (AT_FLOOR)

## Diagnosis

The oracle computes the complete MobileBERT embedding-add-affine scope as one row/hidden Triton writer, including the sliced dynamic token embedding, the all-zero token-type embedding folded to direct loads, the per-hidden multiply/add affine, and three aliasing final `[32768, 512]` views over one backing buffer. Tuned Inductor already reaches the same mandatory memory-traffic envelope for this captured slice/full/embedding/broadcast/add/mul/add/view graph. The ratio of 1.001x confirms both are at the identical bandwidth floor.

- Inductor kernel count: 1
- Oracle kernel count: 1

## Commands

```bash
python repros/canonical/pointwise_bcd0a32063dd/oracle_embedding_affine_aliases.py --check
python repros/canonical/pointwise_bcd0a32063dd/oracle_embedding_affine_aliases.py --bench
```
