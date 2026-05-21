# Repro Quality Progress

Branch: `repro-correctness-integrated`
Base: `origin/main` at `afd3663f`

## Priority Tasks

### 1. Fix partitioner: no horizontal fusion of independent chains
- [x] Use PyTorch `CapabilityBasedPartitioner(skip_horizontal_fusion=True)` from pytorch/pytorch#170191.
- [x] Main has removed the temporary custom connected-components split.
- [x] `capture_hook.py` stays aligned with main's partitioner path; this branch does not reintroduce the old splitter.
- [x] `scripts/test_bench_accounting.py` covers independent branch splitting and producer -> `getitem` -> consumer retention when the installed PyTorch has pytorch/pytorch#170191.

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
- [x] `capture_hook.py` emits compact `Perm(N)` shape configs for inverse-permutation index tensors instead of falling back to blanket `Index(N)`.
- [x] Repaired the four vLLM MoE routing repros that needed permutation inputs.
- [x] Ran read-only vLLM bounds validation: 69 ok, 0 failed, 0 repairable, 0 updated.

### 5. Corpus report and cleanup
- [x] Added `scripts/report_repro_corpus.py` for read-only counts of versions, missing shapes, model manifest references, full graph availability, and safe deletion order.
- [x] Deleted 10 explicit canonical dirs that no manifest referenced.
- [x] Removed stale flat and directory manifests with missing canonical refs; kept saved `full_graph_*.py` source artifacts.
- [x] Merged the 10 duplicate `pattern_hash` canonical pairs into one directory per hash, preserving all shape variants in `shapes.txt`.
- [x] `merge_captures.py` now reuses an existing canonical directory by `pattern_hash` so future merges do not create duplicate-hash dirs under different op labels.
- [x] Added `scripts/test_repro_corpus_invariants.py` to fail on duplicate hashes, missing `_shapes_config`, stale manifest refs, or unreferenced canonical hashes.
- [ ] Remove stale results/shapes artifacts only as explicit reviewed paths, not wildcard cleanup.

## Current Corpus Report

Compared with `origin/main` at `afd3663f`, this branch reduces the corpus from
1709 canonical dirs / 1699 unique hashes / 732 missing manifest refs to:

- Canonical repros: 1689
- Current v2 repros: 1689
- Missing `_shapes_config`: 0
- Manifest files: 439
- Unique manifest pattern refs: 1689
- Manifest refs missing canonical: 0
- Canonical hashes unreferenced by manifests: 0
- Invalid JSON files: 0
- Saved full graph source dirs: 233
- Saved full graph files: 617
- Known duplicate canonical pattern hashes: 0

## Verification

Latest passing checks:

```bash
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_bench_accounting.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_repro_versioning.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/validation/test_update_bounds.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_recapture_full_graphs.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_report_repro_corpus.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_repro_corpus_invariants.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/report_repro_corpus.py --max-examples 20
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_repro_format.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_merge_captures.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/test_validate_eager.py
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python -m py_compile capture_hook.py merge_captures.py repro_harness.py scripts/upgrade_repros.py scripts/test_repro_format.py scripts/test_repro_versioning.py scripts/validation/update_bounds.py scripts/validation/test_update_bounds.py scripts/recapture_full_graphs.py scripts/test_recapture_full_graphs.py scripts/report_repro_corpus.py scripts/test_report_repro_corpus.py scripts/test_repro_corpus_invariants.py scripts/validate_eager.py scripts/test_validate_eager.py scripts/test_merge_captures.py scripts/test_bench_accounting.py
```

Focused GPU validation:

```bash
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/validate_eager.py <10 merged duplicate-hash canonical dirs> --all-shapes --gpus 0 --max-workers 1 --timeout-s 120 --output /tmp/better-benchmark-collision-eager.json
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/validate_eager.py <4 vLLM Perm-fixed canonical dirs> --all-shapes --gpus 1 --max-workers 1 --timeout-s 120 --output /tmp/better-benchmark-vllm-perm-eager.json
TORCH_NATIVE_SKIP_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 python scripts/validation/update_bounds.py <67 vLLM canonical dirs> --all-shapes --device cuda --timeout-s 120 --output /tmp/better-benchmark-vllm-bounds.json
```

Latest results: 1689/1689 canonical repros pass format checks, duplicate hashes are 0, focused eager validation is 18 ok / 0 failed, and vLLM bounds validation is 69 ok / 0 failed.
