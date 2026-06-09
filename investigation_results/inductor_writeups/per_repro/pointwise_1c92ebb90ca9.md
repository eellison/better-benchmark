# pointwise_1c92ebb90ca9 — Reformer head layout

## Summary
- **Repro**: pointwise_1c92ebb90ca9
- **Ratio**: 0.964 (oracle 8.06us vs compile 7.78us)
- **Status**: AT_FLOOR

## Result
Inductor matches or slightly beats the oracle (compile is 3.6% faster).
The compiled Reformer head layout transformation is already optimal.
