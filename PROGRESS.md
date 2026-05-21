# Repro Quality Progress

Branch: `repro-correctness-20260520`
Base: `origin/main` at `eac9147f`

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

### 2. Recapture from saved full graphs first
- [x] Added `scripts/recapture_full_graphs.py` to replay saved `repros/models/**/full_graph_*.py` artifacts through the fixed capture hook.
- [x] Generated repro validation is enabled by default; `--no-validate` is explicit.
- [x] Fail-fast summaries count attempted graphs only.
- [ ] Run reviewed recapture into a scratch canonical root and compare report output before replacing canonical repros.
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
- [ ] Delete old canonical repros only after replacement captures validate and the report shows no missing manifest references.
- [ ] Remove stale results/shapes artifacts only as explicit reviewed paths, not wildcard cleanup.

## Current Corpus Report

- Canonical repros: 1499
- Current v2 repros: 374
- Unversioned repros: 1125
- Missing `_shapes_config`: 120
- Manifest files: 213
- Unique manifest pattern refs: 1223
- Manifest refs missing canonical: 0
- Directory manifest model dirs with `full_graph_*.py`: 156/156
- Manifest-backed full graph files: 387
- Raw `full_graph_*.py` files under `repros/models`: 394

## Verification

Passing:

```bash
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_bench_accounting.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_repro_versioning.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/validation/test_update_bounds.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_recapture_full_graphs.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_report_repro_corpus.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python -m py_compile capture_hook.py repro_harness.py scripts/upgrade_repros.py scripts/test_repro_format.py scripts/test_repro_versioning.py scripts/validation/update_bounds.py scripts/validation/test_update_bounds.py scripts/recapture_full_graphs.py scripts/test_recapture_full_graphs.py scripts/report_repro_corpus.py scripts/test_report_repro_corpus.py
```

Known current corpus blocker:

```bash
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_repro_format.py --quick
```

This still fails on 120 existing repros missing `_shapes_config`; that should be resolved through validated recapture/upgrade, not by mass editing old corpus files.
