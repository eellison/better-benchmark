# pointwise_13bd7be3bbab — mT5 attention mask

## Summary
- **Repro**: pointwise_13bd7be3bbab
- **Ratio**: 1.009 (oracle 6.91us vs compile 6.98us)
- **Status**: AT_FLOOR

## Result
Inductor matches the oracle performance within noise (0.9% gap).
The compiled attention mask computation is already at parity with the hand-written oracle.
