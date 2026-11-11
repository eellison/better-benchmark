# MM pattern corpus

541 repros / 1265 shape points / 15597 source occurrences.

- `prologue`: 30 repros / 73 shape points
- `epilogue`: 415 repros / 1029 shape points
- `horizontal`: 91 repros / 156 shape points
- `horizontal_combine`: 5 repros / 7 shape points

Regenerate with `python scripts/extract_matmul_corpus.py`; `python scripts/validate_matmul_repros.py` runs every shape point eagerly on a GPU. Prologue/epilogue repros are normal BB partitions that absorbed one producer MM (epilogue, tried first) or one consumer MM (prologue), trimmed to the work that depends on the MM, with non-MM work besides views (a cast or copy counts), at most 31 non-MM compute ops, no CPU tensor inputs and, for epilogues, at most 4 extra input tensors. Horizontal windows group 2-3 independent MMs that share an operand. Repros return values, not views of them. Every repro is one pattern hash, which ignores scalar literals as in repros/: literal variants are one repro with the first captured code. There is no repro or shape-point cap. `canonical/<category>_<mm ops>_<hash>/` is flat like `repros/canonical/`, so the repro tools (`validate_eager.py`, `bench_parallel.py`, `validate_corpus_invariants.py --corpus-root mm_repro`) run on it. See [docs/matmul_partitioning.md](../docs/matmul_partitioning.md).

[Report](report.json)
