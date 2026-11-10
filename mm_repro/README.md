# MM pattern corpus

1179 repros / 1904 shape points / 16802 source occurrences.

- `prologue`: 16 repros (one per family; 16 before dedup) / 48 shape points
- `epilogue`: 1102 repros (one per family; 1248 before dedup) / 1753 shape points
- `horizontal`: 56 repros (one per family; 100 before dedup) / 96 shape points
- `horizontal_combine`: 5 repros (one per family; 5 before dedup) / 7 shape points

Regenerate with `python scripts/extract_matmul_corpus.py`. Prologue/epilogue use whole normal BB partitions with producer (epilogue) absorption first and one MM per augmented partition. Epilogues have no reduction-count limit. Horizontal windows retain their existing policy. Each family exports one scalar-preserving representative and only its compatible input points. There is no corpus-count cap.

[Report](report.json)
