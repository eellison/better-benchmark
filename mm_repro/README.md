# MM pattern corpus

529 repros / 1177 shape points / 14751 source occurrences.

- `prologue`: 105 repros (one per family; 109 before dedup) / 408 shape points
- `epilogue`: 345 repros (one per family; 419 before dedup) / 650 shape points
- `horizontal`: 74 repros (one per family; 118 before dedup) / 112 shape points
- `horizontal_combine`: 5 repros (one per family; 5 before dedup) / 7 shape points

Regenerate with `python scripts/extract_matmul_corpus.py`. Prologue/epilogue use whole normal BB partitions with pointwise-consumer absorption first and one MM per augmented partition. Epilogues have no reduction-count limit. Horizontal windows retain their existing policy. Each family exports one scalar-preserving representative and only its compatible input points. There is no corpus-count cap.

[Report](report.json)
