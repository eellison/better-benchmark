# MM pattern corpus

585 repros / 1019 shape points / 13275 source occurrences.

- `prologue`: 49 repros (one per family; 107 before dedup) / 91 shape points
- `epilogue`: 284 repros (one per family; 518 before dedup) / 602 shape points
- `prologue_epilogue`: 79 repros (one per family; 112 before dedup) / 104 shape points
- `horizontal`: 74 repros (one per family; 118 before dedup) / 112 shape points
- `horizontal_combine`: 5 repros (one per family; 5 before dedup) / 7 shape points
- `vertical`: 94 repros (one per family; 102 before dedup) / 103 shape points

Regenerate with `python scripts/extract_matmul_corpus.py`. Scalar values, input domains, strides, offsets and output aliases are preserved. Alignment/copy-only and infinity-mask regions are excluded by the partitioner. There is no corpus-count cap.

[Report](report.json)
