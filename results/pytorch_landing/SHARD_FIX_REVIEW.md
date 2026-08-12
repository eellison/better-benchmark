# Adversarial review — repro `--all-shapes` (dir,shape) sharding fix

Reviewed: uncommitted working-tree change to `scripts/bench_parallel.py` (+219/-22)
plus `tests/test_oracle_sharding.py` (+149). Backups at
`/tmp/scratch_space/repro_shard_fix.diff` and `…/bench_parallel_with_shardfix.py`.

VERDICT (up front): **Correct + safe to commit as-is.** No correctness bugs
found. The flagged `_split_task_key` is NOT a duplicate of
`_split_shape_task_key` (different process/scope — see (a)). One minor,
optional cleanup is noted. Targeted both-ways bench: schema byte-identical,
numbers match within 0.27% (bench noise). The running walk was not disturbed;
nothing committed/reverted/reset.

---

## (a) Reuse verdict — is `_split_task_key` a duplicate?

**No — they live in two different processes and cannot share a symbol.**

- `_split_shape_task_key` (parent, module scope, line 104) — used by the
  orchestrator/regroup code in the parent Python process.
- `_split_task_key` (line 2298+) is **emitted into the generated repro
  *worker* subprocess script** (`_persistent_worker_script`, an f-string
  template). The worker is a standalone `python -c` child that imports nothing
  from `bench_parallel`; it only has the constants the template interpolates
  (`SHAPE_SEP`, `DEFAULT_SHAPE_TOKEN`). So it genuinely needs its own splitter.
  The bodies are equivalent (`rsplit(SEP, 1)` → `(path, label|None)`), but this
  is unavoidable cross-process code, not removable duplication.

  Consistency note (not a bug): the existing **oracle** worker template does
  the same split *inline* (`if SHAPE_SEP in line: … line.rsplit(SHAPE_SEP, 1)`,
  lines 2235–2236) instead of a named helper. So the new repro worker is
  slightly *better* factored than the oracle worker, not worse. Optional polish:
  the oracle worker could adopt the same `_split_task_key` helper for symmetry —
  but that's gold-plating and out of scope for this fix.

- **Regroup vs oracle regroup:** `_regroup_sharded_repro_results` does NOT
  duplicate the oracle path. The oracle path regroups *and flattens into a
  different schema* (`_aggregate_oracle_timings` → flat shape-hash-keyed timings
  for `model_graph_accounting`). The repro path must preserve the
  **per-repro-path, shape-labels-nested** schema that `bench_report.py` /
  `--update-perf` / `--merge-into` consume — a genuinely different output
  contract, so a separate (simple, 8-line) regroup is correct. No shared
  regroup to reuse. `_regroup_sharded_oracle_failures` is oracle-dir-keyed;
  the repro failure regroup has different semantics (partial-success handling,
  below), so it is also correctly separate.

`_make_shape_task_key`, `_split_shape_task_key`, `_SHAPE_TASK_SEP`,
`_DEFAULT_SHAPE_TOKEN`, `load_shape_configs` are all reused (not re-implemented)
in the parent. Good.

## (b) Correctness — bugs found: NONE. Paths traced:

1. **Round-trip / keying.** Worker stamps `results["_repro"] = line` with the
   FULL sharded key; parent validates `_repro == str(repro_path)` (full key,
   line 1993) and stores `all_results[result["repro"]]` / `failures[...]` by the
   full key. The suffix is stripped ONLY at the actual module load
   (`_split_task_key` → `spec_from_file_location(file_path)`); the prefetch
   cache is keyed by the full key on both store and pop. Consistent end-to-end.

2. **Dynamo-reset-per-shape (the contamination fix) PRESERVED.** The per-shape
   loop body (`bench_one`, lines 2700–2787) is byte-identical between sharded
   and un-sharded — same `torch._dynamo.reset()` + fresh `torch.compile` per
   shape, same `inductor_metrics.reset()`, same coord-descent toggling, same
   warmup/N_ROUNDS/`do_bench(return_mode="min")`. Sharding only changes which
   labels populate `shape_items`; it never touches WHAT is measured.

3. **`__default__` path.** A no-config repro yields one
   `…::SHAPE::__default__` task; the worker routes it to
   `shape_items = [(None, None)]` → `make_inputs` default branch → label
   `"default"`, identical to the un-sharded `all_shapes and configs` =False
   fallback. (No no-config dir exists in the current corpus to bench live, but
   the logic is verified by read + by `test_expand_repro_shape_tasks` which
   asserts the `__default__` token is emitted for empty configs.)

4. **Failure regroup attribution** (`_regroup_sharded_repro_failures`):
   - All shapes of a repro failed → ONE entry under the bare repro.py path,
     annotated `failed_shapes=[…]`. ✓ (test_regroup_repro_failures_all_shapes_failed)
   - Partial: some shapes succeeded → success payload left untouched; the
     failed shape recorded under its SHAPE-qualified key so it's accounted/
     resumable without colliding with the bare-path success. ✓
     (test_regroup_repro_failures_partial_keeps_success_byte_identical)
   - Bare (un-sharded) failures pass through. ✓
   Shape-qualified failure keys land under reserved `__failures__` (nested), so
   `_success_items` (top-level only) never mistakes them for results.

5. **Summary recount** (lines 1738–1745): after regroup, `done = len(all_results)`
   (repros, not shape-tasks) and `failed` counts only bare-path failures
   (label is None), correctly NOT double-counting a partially-failed repro that
   has surviving shapes. Verified live: a 3-shape dir reports `ok:1`, not `ok:3`.

6. **Incremental snapshot** (lines 1691–1693): regroups before the every-50
   write so mid-run JSON uses the same bare-keyed schema. Passes the *already
   regrouped* `snap_results` as the `all_results` arg to the failure regroup —
   correct, since `succeeded` must be the bare-path set there.

## (c) Scope-creep note — `_sort_shape_tasks_big_dirs_first` (LPT/biggest-first)

This is **not new behavior** introduced by this fix. The identical
biggest-dirs-first sort already existed inline on the oracle path (the diff
*extracts* it into a shared helper and reuses it for both paths — a net
de-duplication). The ordering is a pure scheduling hint (stable sort by
`-shape_count, path`); it cannot change any measured number, only worker
drain order. Correct and harmless. `test_sort_shape_tasks_big_dirs_first`
covers it. Mild scope note: extracting the helper touched the oracle path, but
it's behavior-preserving refactor, not a semantic change.

## (d) Targeted numbers-match (one multi-shape dir, both ways)

Dir: `repros/canonical/amax_sum_ac4bab3e35d2` (3 shapes). GPU 0 only,
`--workers-per-gpu 1`. OLD = HEAD-committed un-sharded bench_parallel (verified
`grep -c shard_repros == 0`); NEW = working-tree sharded version.

- **Task granularity:** OLD ran "1 repro"; NEW ran "3 repros" (one per shape)
  and regrouped back to 1 repro.py key. ✓
- **Schema:** top-level keys identical (`_metadata`, `__summary__`, the single
  bare `…/repro.py` key); per-shape metric key sets identical
  (`compiled_us, coord_descent_us, gap_cd, gap_default, memcopy_sol_us,
  n_kernels, total_bytes`); no `__failures__`; `__summary__` = `total:1, ok:1,
  failed:0` in both. ✓
- **Numbers** (max relative deviation across all 3 shapes × 7 metrics):
  `compiled_us` ≤0.06%, `coord_descent_us` ≤0.21%, `memcopy_sol_us` ≤0.27%,
  `gap_*` ≤0.26%; `total_bytes` and `n_kernels` **byte-identical**. All within
  bench noise (threshold 8%). ✓
- 25/25 unit tests pass (`tests/test_oracle_sharding.py`), incl. the 8 new
  repro-sharding tests.

Outputs kept at `/tmp/scratch_space/shard_verify/{OLD_unsharded,NEW_sharded}.json`.

## (e) VERDICT

**Correct + safe to commit as-is.** Sharding changes only task granularity and
worker drain order; the measured numbers, output schema, perf.json/merge
contract, and the dynamo-reset-per-shape methodology are all preserved. No
duplication that should be consolidated (the two splitters are unavoidably
cross-process). No correctness bug in the regroup / `__default__` /
partial-failure / summary-recount paths.

Optional, non-blocking polish (do NOT gate the commit): (1) consider having the
oracle worker template reuse the new `_split_task_key` helper for symmetry;
(2) test coverage for the `__default__` end-to-end worker dispatch is only
indirect (no no-config dir in corpus) — fine, but a synthetic worker-level test
would close the last gap.

---

**Walk safety confirmed:** ran ONLY on GPU 0 (idle at run time), tiny dirs;
did not commit, revert, or reset. Working tree still shows `M
scripts/bench_parallel.py` with the unchanged 345-line diff. Temp HEAD copy
placed in `scripts/` for the baseline run was removed (the pre-existing
`scripts/_bench_parallel_orig_verify.py`, 14:33, is not mine — left untouched).
