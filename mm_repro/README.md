# MM pattern corpus

498 repros / 945 shape points / 9305 source occurrences.

- `prologue`: 25 repros (one per family; 25 before dedup) / 36 shape points
- `epilogue`: 412 repros (one per family; 412 before dedup) / 801 shape points
- `horizontal`: 56 repros (one per family; 93 before dedup) / 101 shape points
- `horizontal_combine`: 5 repros (one per family; 5 before dedup) / 7 shape points

Regenerate with `python scripts/extract_matmul_corpus.py`. Prologue/epilogue use whole normal BB partitions with producer (epilogue) absorption first and one MM per augmented partition. Epilogues have no reduction-count limit. Horizontal windows retain their existing policy. Pattern hashes ignore scalar literals, as in repros/: literal variants are one repro with the first captured code. Each family exports one representative. There is no corpus-count cap.

[Report](report.json)
