# Adversarial Review R1: oracle_impl dispatch (run inline, Bedrock killed the agent)

Probed: parser vs all 1482 corpus signatures, empty/S-only signatures, kwarg
collisions, downstream JSON parsers, worker module-keying, f8 dtypes,
bench_oracle callable audit (20 sampled), S/T positional interleave,
all_shapes path.

## Findings

1. **MAJOR — empty / S()-only signatures never match (29 repros).**
   29 repros have `_shapes_config` with zero tensor entries: 13x `"()"` and
   16x S()-only like `"(S([1, -1, 512, 512]))"` (all inputs are shape params
   / scalars). `parse_shapes_signature` returns `()`, and
   `OracleRegistry._normalize_shape(())` mis-normalizes it to `((),), False`
   (a "key-shape" registration with an empty key). `_signature_matches` then
   requires `bool(actual_shapes)` — actual is also `()` — so it can NEVER
   match: migrating these 29 repros would make their oracles permanently
   NO_ORACLE_FOR_SHAPE. Verified by probe (raises today).
   Fix: empty parsed signature -> full-signature-of-zero-tensors
   (`shape=(), full_sig=True`), matching empty `get_shape_key` output.

2. **MINOR — f8 dtypes missing from the honesty map.** 95 repros mention
   float8 in repro.py. No `_shapes_config` string currently uses an f8 token
   (parser ns would NameError if one did), but the `_DT` map in select()
   falls back to `str(t.dtype)` so worst case is a cosmetic mismatch string,
   and parse ns should add f8 names defensively for future shapes.txt lines.

3. **OK — downstream parsers safe.** bench_oracles_parallel.py,
   measure_all_oracles.py, build_gap_tracker.py: zero hard `["oracle_us"]`
   key accesses; all use .get(). NO_ORACLE_FOR_SHAPE lines (no oracle_us/
   ratio keys) flow through as unmeasured entries.

4. **OK — no worker module collision.** bench_oracles_parallel.py runs each
   oracle as `subprocess [sys.executable, oracle_path, "--bench"]` — one
   process per oracle, fn.__module__ == "__main__" is process-private.
   (If a future runner imports multiple oracle files in-process via
   importlib, distinct module names keep registries separate.)

5. **OK — bench_oracle callable audit.** 20/20 sampled oracle files pass
   `oracle_forward` itself (not a wrapper) to bench_oracle, so module-keyed
   resolution works for the mechanical migration.

6. **OK — no S/T positional misalignment.** 0 signatures have S() entries
   before the last T(); S params always trail tensors, so skipping them
   aligns with get_shape_key (tensors-only) positionally.

7. **OK — kwarg reserved names coexist** (description= + BLOCK= verified).
   Residual NIT: an impl wanting a literal kwarg named "hardware"/"shapes"/
   "description" can't get it through oracle_impl; acceptable, documented.

8. **OPEN (design question, not a bug) — all_shapes semantics.** A migrated
   oracle registered at 1 shape over a 10-line shapes.txt yields 1 measured
   line + 9 NO_ORACLE_FOR_SHAPE lines. Loud-and-correct per design; flagged
   for the user to confirm vs an UNTUNED_SHAPE measure-anyway mode.
