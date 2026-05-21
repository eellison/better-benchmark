# Repro Quality Progress

Branch: `repro-correctness-integrated`
Base: `origin/main` at `d70c6650`

## Priority Tasks

### 1. Fix partitioner: no horizontal fusion of independent chains
- [x] Split `CapabilityBasedPartitioner` output into real data-dependent components before extraction.
- [x] Treat `getitem`, view-like ops, `alias`, and `detach` as transparent adapters.
- [x] Pull transparent input adapters back into partitioner regions before hashing, so external `getitem` selectors are not lost.
- [x] Keep producer -> `getitem` -> consumer chains together.
- [x] Do not let external tuple `getitem`s glue sibling chains together.
- [x] Encode structural selector ints in the DAG signature, so `x[0]` and `x[1]` do not dedupe as the same pattern.
- [x] Prune unused transparent descendants from tuple-producing compute regions.
- [x] Tests cover independent chains, alias/detach branch points, getitem dependencies, getitem hash identity, and unused getitem pruning.

### 2. Recapture all models with fixed partitioner
- [x] Main has been repartitioned/recaptured from saved full graphs.
- [x] Added `scripts/recapture_full_graphs.py` as a validated recapture path with `--no-validate` explicit.
- [x] Fail-fast summaries count attempted graphs only.
- [x] Removed stale model manifests whose pattern refs pointed at deleted pre-recapture canonical hashes.
- [x] Re-ran corpus report and format sweep after integration; canonical repros have zero missing `_shapes_config`.
- [ ] Fall back to source model recapture only for missing/stale full graph artifacts.

### 3. Format versioning + upgrade path
- [x] Added `CURRENT_REPRO_VERSION = 2` and unversioned v0 parsing in `repro_harness.py`.
- [x] `test_repro_format.py` validates explicit markers and real top-level `_shapes_config` assignments.
- [x] `upgrade_repros.py` consumes the shared version and rejects unsupported/future/invalid versions.
- [x] Legacy upgrade fails closed if recapture yields zero or multiple regions.
- [ ] Bulk upgrade only after recaptured replacements validate.

### 4. Fix generated index bounds
- [x] Added `scripts/validation/update_bounds.py` to stress-fill `Index(...)` tensors, classify index-bound failures, binary-search smaller exclusive highs, and optionally rewrite affected shape configs.
- [x] `--all-shapes` validates default `_shapes_config` plus `shapes.txt`; empty shape sets now fail instead of passing silently.
- [x] Tests cover bad `Index(10)` repair, default-only rewrite, shapes.txt rewrite, non-index eager failures, missing labels, and empty shape sets.
- [ ] Run against the vLLM/MoE failures and review any proposed bound rewrites before committing corpus changes.

### 5. Corpus report and cleanup
- [x] Added `scripts/report_repro_corpus.py` for read-only counts of versions, missing shapes, model manifest references, full graph availability, and safe deletion order.
- [x] Deleted 10 explicit canonical dirs that no manifest referenced.
- [x] Removed stale flat and directory manifests with missing canonical refs; kept saved `full_graph_*.py` source artifacts.
- [ ] Remove stale results/shapes artifacts only as explicit reviewed paths, not wildcard cleanup.

## Current Corpus Report

- Canonical repros: 1699
- Current v2 repros: 1699
- Missing `_shapes_config`: 0
- Manifest files: 439
- Unique manifest pattern refs: 1689
- Manifest refs missing canonical: 0
- Canonical hashes unreferenced by manifests: 0
- Invalid JSON files: 0
- Saved full graph source dirs: 233
- Saved full graph files: 617
- Known duplicate canonical pattern hashes: 10

## Verification

Passing before integration:

```bash
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_bench_accounting.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python test_connected_components.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_repro_versioning.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/validation/test_update_bounds.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_recapture_full_graphs.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_report_repro_corpus.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python -m py_compile capture_hook.py repro_harness.py scripts/upgrade_repros.py scripts/test_repro_format.py scripts/test_repro_versioning.py scripts/validation/update_bounds.py scripts/validation/test_update_bounds.py scripts/recapture_full_graphs.py scripts/test_recapture_full_graphs.py scripts/report_repro_corpus.py scripts/test_report_repro_corpus.py
```

Required before merge:

```bash
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/report_repro_corpus.py --max-examples 20
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_repro_format.py --quick
```

The format sweep must pass with zero canonical repros missing `_shapes_config`.

Latest result: passing for all 1699 canonical repros.
