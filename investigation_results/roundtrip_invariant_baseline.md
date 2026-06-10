# Full-Graph Round-Trip Invariant Baseline (2026-06-10)

Run: `CUDA_VISIBLE_DEVICES="" python scripts/validate_corpus_invariants.py
--full-graph-roundtrip --sample 8 --json investigation_results/roundtrip_invariant_baseline.json`
(seed=0, deterministic; 4m31s wall on CPU, no GPU work — all input
construction and re-tracing under FakeTensorMode with patched
`torch.cuda.is_available`).

Per-model raw results: `investigation_results/roundtrip_invariant_baseline.json`.

## Invariants

- **A (input round-trip)**: `full_graph_harness.load_full_graph_definition` +
  `make_tensor_from_spec` succeeds for every input spec / tensor attr of every
  sampled `full_graph_*.py`. Failure names the placeholder and reason.
- **B (partition determinism)**: `capture_hook.get_fusion_partitions` run twice
  on the same re-traced GraphModule yields identical pattern-hash multisets.
- **C (partition round-trip)**: pattern-hash set derived by re-tracing the
  saved artifact (make_fx fake mode + the capture pipeline's own partitioner /
  hasher) matches `manifest.json` patterns. Mismatches explained purely by
  view-op spelling drift during re-tracing are a KNOWN class (does not fail);
  anything else is a hard mismatch (exit 1).

## Corpus coverage

| Class | Model dirs |
|---|---|
| Have `full_graph_*.py` (round-trip applies) | **197** (genai=8, hf=44, timm=36, torchbench=104, vllm=5) |
| Manifest-only, no `full_graph_*.py` (KNOWN class: round-trip N/A, need recapture) | **404** |

The 404 manifest-only dirs are the pre-graph_dir captures (old layout, e.g.
`hf/infer/hf_AlbertForMaskedLM_infer_000/`); they carry only
`manifest.json` pattern lists. This matches the existing memory note that 404
model dirs need recapture (dtype policy). They are reported, not failed.

A second structural finding: of the 197 dirs WITH full graphs, only the 36
timm dirs also have `manifest.json` — so invariant C is currently only
exercisable on timm. The other 161 (genai/hf/torchbench/vllm new-layout dirs)
have graphs but no manifest ("C:n/a(no-manifest)" below); their pattern lists
live only in the old-layout manifest-only sibling dirs.

## Results per suite (8 models/suite sampled, seed=0; 37 dirs, 82 graphs)

| Suite | Models | Graphs | A: input round-trip | B: partition determinism | re-trace | C: partition round-trip |
|---|---|---|---|---|---|---|
| genai | 8 | 12 | PASS (0 fail) | PASS (0 fail) | PASS | n/a — no manifest (8/8) |
| hf | 8 | 31 | PASS (0 fail) | PASS (0 fail) | PASS | n/a — no manifest (8/8) |
| timm | 8 | 12 | PASS (0 fail) | PASS (0 fail) | PASS | PASS, all 8 spelling-only (known class) |
| torchbench | 8 | 14 | PASS (0 fail) | PASS (0 fail) | PASS | n/a — no manifest (8/8) |
| vllm | 5 (all) | 13 | PASS (0 fail) | PASS (0 fail) | PASS | n/a — no manifest (5/5) |

**Overall: PASS — 0 A violations, 0 B violations, 0 re-trace failures,
0 hard C mismatches.** Exit code 0.

## Diagnoses

### A — input round-trip: clean
Every sampled artifact (including the 2-billion-element Mistral-7B vllm graph,
constructed fake) reconstructs inputs from sidecar `.meta.json` or forward
annotations. No placeholder failures. The earlier backfill of `.meta.json`
sidecars (commits 1980e3480 etc.) appears complete for the sampled set.

### B — partition determinism: clean
`CapabilityBasedPartitioner` + `is_fusible_node` + `skip_horizontal_fusion`
is deterministic across repeated runs on the same gm in every sampled graph
(largest: AlbertForMaskedLM train, 106 partitions).

### C — partition round-trip: all mismatches are re-trace spelling drift (timm only)
All 8 timm sampled models show the same known class, in three tiers:

1. **reshape→view re-spelling** (most common, e.g. `e262d057f3c9`
   [`reshape+sum`] missing vs `22b5cd24890b` [`view+sum`] extra — the exact
   pair in every timm train model sampled). Root cause: compile_fx runs
   `view_to_reshape` BEFORE the post-grad capture hook, so manifests/canonical
   hashes were minted with reshape spelling; make_fx re-tracing decomposes
   back to view. The concurrent `canonicalize_for_hash` fix (commit 6ee455f4a)
   closes this fork at hash time going forward, but existing manifest +
   canonical-dir hashes were minted pre-fix, so the validator classifies these
   by comparing non-transparent op multisets instead of failing.
2. **clone/_unsafe_view layout-copy placement drift** (swin
   `91fc2812ef17`, `deb7c9191e39`): saved trace spells
   reshape-of-non-contiguous as `clone + _unsafe_view`; re-trace absorbs or
   re-places the copy in a neighboring partition.
3. **pure layout/copy partitions** (deit `baa1198c62c0` =
   `permute+clone+_unsafe_view+view`): the partition has no compute at all
   modulo copies/views; its boundary is entirely a trace-spelling artifact.

No hard (non-spelling) C mismatch in the sample.

### Known classes summary

| Known class | Count in sample | Action |
|---|---|---|
| manifest-only model dirs (no full graphs) | 404 (corpus-wide) | recapture (already tracked) |
| full-graph dirs without manifest.json | 161 (corpus-wide), 29/37 sampled | backfill manifests from old-layout dirs or repartition |
| C spelling-only drift (reshape/view, clone/_unsafe_view) | 8/8 timm models | closes when manifests+canonical re-minted with `canonicalize_for_hash` hashes |

## Where the validation lives

- `roundtrip_validation.py` — invariants A/B/C, spelling classification,
  validate-before-write gate (`run_write_gate`).
- `scripts/validate_corpus_invariants.py --full-graph-roundtrip
  [--all|--sample N|--seed S|--json PATH]` — exit 1 on any A/B violation,
  re-trace failure, or non-spelling C mismatch.
- Write gate wired into `capture_hook._CaptureState.process_graph` and
  `extract_reductions.py`: every newly written `full_graph_NNN.py` is
  reloaded, validated (A + C vs the live gm), and its sidecar stamped
  `"roundtrip": "ok" | "failed: <reason>"`. Artifacts are never deleted.
