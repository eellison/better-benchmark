# MM pattern corpus

535 repros / 1001 shape points / 9305 source occurrences.

- `prologue`: 25 repros / 36 shape points
- `epilogue`: 412 repros / 801 shape points
- `horizontal`: 93 repros / 157 shape points
- `horizontal_combine`: 5 repros / 7 shape points

Regenerate with `python scripts/extract_matmul_corpus.py`; `python scripts/validate_matmul_repros.py` runs every shape point eagerly on a GPU. Prologue/epilogue repros are normal BB partitions that absorbed one producer MM (epilogue, tried first) or one consumer MM (prologue), trimmed to the work that depends on the MM, with 1 to 31 non-MM compute ops and, for epilogues, at most 4 extra input tensors. Horizontal windows group 2-3 independent MMs that share an operand. Every repro is one pattern hash, which ignores scalar literals as in repros/: literal variants are one repro with the first captured code. There is no repro or shape-point cap. `canonical/<category>_<mm ops>_<hash>/` is flat like `repros/canonical/`, so the repro tools (`validate_eager.py`, `bench_parallel.py`, `validate_corpus_invariants.py --corpus-root mm_repro`) run on it. See [docs/matmul_partitioning.md](../docs/matmul_partitioning.md).

[Report](report.json)
