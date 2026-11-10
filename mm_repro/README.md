# MM pattern corpus

1192 repros / 1918 shape points / 16956 source occurrences.

- `prologue`: 11 repros (one per family; 11 before dedup) / 43 shape points
- `epilogue`: 1102 repros (one per family; 1248 before dedup) / 1756 shape points
- `horizontal`: 74 repros (one per family; 118 before dedup) / 112 shape points
- `horizontal_combine`: 5 repros (one per family; 5 before dedup) / 7 shape points

Regenerate with `python scripts/extract_matmul_corpus.py`. Prologue/epilogue use whole normal BB partitions with producer (epilogue) absorption first and one MM per augmented partition. Epilogues have no reduction-count limit. Horizontal windows retain their existing policy. Each family exports one scalar-preserving representative and only its compatible input points. There is no corpus-count cap.

[Report](report.json)
