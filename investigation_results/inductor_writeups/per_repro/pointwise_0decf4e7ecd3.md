# pointwise_0decf4e7ecd3 — attention Q layout

## Summary
- **Repro**: pointwise_0decf4e7ecd3
- **Ratio**: 1.033 (oracle 5.89us vs compile 6.08us)
- **Status**: AT_FLOOR

## Result
Inductor matches the oracle performance within noise (3.3% gap is below the 5% investigation threshold).
The compiled kernel handles the attention Q layout transformation at near-optimal throughput.
