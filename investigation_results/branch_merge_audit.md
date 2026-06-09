# Branch Merge Audit: investigations-june-2026 -> main

Generated: 2026-06-09
Branch: `investigations-june-2026` (800+ commits ahead of main)
Main state: Clean infra (PRs #57, #58)

Total files changed on branch (excluding `repros/`): 1628
Of those, ~1479 are per-repro writeups in `investigation_results/inductor_writeups/per_repro/`.

---

## SHOULD MERGE TO MAIN

These are stable infrastructure that other sessions/users need.

### Core Harness & Infrastructure

| File | Rationale |
|------|-----------|
| `oracle_harness.py` (782 lines) | Shared oracle infrastructure: load repros, correctness checks, bench with CUDAGraph+interleaved timing, OracleRegistry dispatch. Every oracle imports this. |
| `full_graph_harness.py` (patch) | Bug fix: clamp `high` to dtype max for int8 inputs in `_randint_storage`. Prevents overflow. Already on main, just needs the 10-line patch. |
| `scripts/oracle_template.py` | Standardized oracle template. All new oracles copy this. |
| `scripts/ORACLE_FORMAT.md` | Authoritative standard for oracle format, timing rules, numeric rules. |
| `scripts/oracle_dispatch_design.md` | Design doc for registration-based multi-impl dispatch (implemented in oracle_harness). |
| `scripts/validate_oracles.py` (442 lines) | Validates oracle correctness and detects scope mismatches. Produces CSV. |
| `scripts/batch_convert_oracles_to_harness.py` (801 lines) | Batch migration tool: convert ad-hoc oracles to standardized harness format. |
| `scripts/backfill_integer_constraints.py` (558 lines) | Backfills .meta.json bounds for full graphs with integer inputs. Makes graphs benchmarkable. |
| `scripts/retroactive_bounds_inference.py` | Source-level bounds inference for full_graph files. Used by backfill script. |
| `scripts/model_attribution.py` (690 lines) | Per-model performance attribution: maps model graphs -> canonical repros -> compile/oracle times. |
| `scripts/model_graph_accounting.py` (971 lines) | FX graph traversal: classifies ops as fusible/non-fusible, builds compute structure accounting. |
| `scripts/compare_sweeps.py` (94 lines) | Compare two sweep JSONs (baseline vs treatment). Clean utility. |
| `scripts/export_investigation_tables.py` (361 lines) | Export sweep data to CSV/SQLite tables. Drives downstream analysis. |
| `scripts/measure_all_oracles.py` (337 lines) | Batch-measure all oracles, parse timing, build gap tracker. |
| `scripts/build_gap_tracker.py` (291 lines) | Build oracle_vs_compile_tracker.csv from measurement logs. |
| `scripts/export_realistic_floors.py` (232 lines) | Export heuristic floor estimates per SOL-gap candidate. |
| `scripts/_single_repro_runner.py` (54 lines) | Helper: run one repro, compile, time, print JSON. Used by rerun_oom_singles. |
| `scripts/rerun_oom_singles.py` (121 lines) | Re-run OOM'd repros individually. |

### Reference Kernels

| File | Rationale |
|------|-----------|
| `optimal_kernels/softmax_fwd_8192x262144.py` | Hand-optimized Triton kernel with full root-cause analysis of Inductor's gap. Reference implementation. |
| `optimal_kernels/softmax_bwd_8192x262144.py` | Backward pass optimal kernel. |
| `optimal_kernels/cross_entropy_fwd_8192x262144.py` | Cross-entropy reference. |
| `optimal_kernels/layernorm_fwd_1152000x512.py` | LayerNorm reference. |
| `optimal_kernels/rmsnorm_fwd_1152000x512.py` | RMSNorm reference. |

### Key Documentation (authoritative, not session-specific)

| File | Rationale |
|------|-----------|
| `AGENT_INSTRUCTIONS.md` | Critical context for any agent working on this repo. Defines allowed fixes, investigation workflow, config knobs. |
| `INVEST_INSTRUCTIONS.MD` | Collaboration guide: what this repo is, current state, implemented optimizations, what's left. |
| `ORACLE_GAP_CLOSING_PLAYBOOK.md` | Methodology for iterating on oracle gaps. |
| `investigation_results/kernel_family_classification.md` | Authoritative classification of all 1090 SOL-gap repros into families. |
| `investigation_results/kernel_priority_by_model_impact.md` | Priority ranking by model-level impact. |
| `investigation_results/optimization_status_june2026.md` | Status summary of all optimizations. |

---

## KEEP ON BRANCH (investigation-specific, ephemeral)

### Per-Repro Writeups (~1479 files)

All files under `investigation_results/inductor_writeups/per_repro/` (amax_sum_*, mean_*, pointwise_*, sum_*, var_mean_* writeups). These are investigation artifacts documenting individual repro analysis. They should stay on the investigation branch as reference but don't belong on main.

### Sweep Result JSONs (root-level, large)

| File | Rationale |
|------|-----------|
| `sweep_3config_interleaved.json` | Session-specific sweep data |
| `sweep_3config_with_stash.json` | Session-specific |
| `sweep_3config_with_stash_summary.txt` | Session-specific |
| `sweep_all_fixes.json` | Session-specific |
| `sweep_fixed_bench.json` | Session-specific |
| `sweep_pr184905_all_knobs.json` | Session-specific |
| `sweep_pr184905_all_knobs_v2.json` | Session-specific |
| `sweep_pr184905_baseline.json` | Session-specific |
| `sweep_pr184905_combo_perblock.json` | Session-specific |
| `sweep_pr184905_combo_perblock_comparison.txt` | Session-specific |
| `sweep_pr184905_comparison.txt` | Session-specific |
| `sweep_pr184905_multi_kernel.json` | Session-specific |
| `sweep_pr184905_multi_kernel_comparison.txt` | Session-specific |
| `sweep_scalar_acc_fixed.json` | Session-specific |
| `sweep_scalar_acc_only.json` | Session-specific |
| `sweep_scalar_acc_targeted.json` | Session-specific |
| `sweep_tiling_only_full.json` | Session-specific |
| `sweep_tiling_scores_full.json` | Session-specific (48K lines) |
| `sweep_with_scalar_acc.json` | Session-specific |
| `algebraic_elim_results.json` | Session result |
| `all_regressions.json` | Session result |
| `combined_allfixes_10repro_results.json` | Session result |
| `measurement_online_softmax.json` | Session result |
| `online_softmax_heuristic_data.json` | Session result |

### Investigation Queue CSVs and Tracking

| File | Rationale |
|------|-----------|
| `investigation_results/RUNNING_INVESTIGATIONS.csv` | Session-specific queue |
| `investigation_results/RUNNING_INVESTIGATIONS.md` | Session-specific |
| `investigation_results/SESSION_SUMMARY.md` | Session-specific |
| `investigation_results/inductor_optimization_active_queue.md` | Ephemeral session queue |
| `investigation_results/inductor_optimization_per_repro_queue.csv` | Session queue |
| `investigation_results/inductor_optimization_priority_queue.csv` | Session queue |
| `investigation_results/inductor_optimization_tracker.csv` | Session tracker |
| `investigation_results/inductor_target_configs.csv` | Session-specific |
| `investigation_results/oracle_gap_closure_queue.csv` | Session queue |
| `investigation_results/oracle_kernel_work_queue.csv` | Session queue |
| `investigation_results/oracle_optimization_queue.csv` | Session queue |
| `investigation_results/oracle_priority_worklist.csv` | Session queue |
| `investigation_results/realistic_floor_worklist.csv` | Session queue |
| `investigation_results/bad_oracles_need_rewrite.txt` | Session-specific triage |

### Session-Specific Investigation Docs

| File | Rationale |
|------|-----------|
| `PROGRESS.md` | Session log (build status, sweep runs, timestamps) |
| `investigation_combo_all_slowdowns.md` | Session writeup |
| `investigation_combo_mechanism.md` | Session writeup |
| `investigation_combo_regressions.md` | Session writeup |
| `investigation_multi_kernel.md` | Session writeup |
| `docs/combo_kernel_pid_subblock_interference.md` | Session investigation doc |
| `docs/combo_kernel_regressions.md` | Session investigation doc |
| `investigation_results/oracle_measurement_logs/` | Raw measurement logs |
| `investigation_results/oracle_batch_*_results.json` | Batch measurement artifacts |
| `investigation_results/model_attribution_sample.md` | One-off sample |
| `investigation_results/scope_mismatch_oracles.md` | Session finding |
| `investigation_results/speedup_attribution_validation.md` | Session validation |

### One-Off Test Scripts (root-level)

| File | Rationale |
|------|-----------|
| `test_tl_trans_perf.py` | One-off perf test |
| `test_transpose_sum_no_atomic.py` | One-off experiment |
| `bench_no_oracle.py` | Ad-hoc benchmark |

---

## MAYBE (useful but needs cleanup before merging)

### Scripts with Hardcoded Paths

| File | Issue |
|------|-------|
| `scripts/build_gap_tracker.py` | Hardcoded `ROOT = Path("/tmp/scratch_space/better_benchmark")` instead of relative |
| `scripts/measure_all_oracles.py` | Same hardcoded ROOT |

These are good infra but need the hardcoded paths fixed to `Path(__file__).resolve().parents[1]` before merging.

### Prototype/Experiment Scripts

| File | Issue |
|------|-------|
| `scripts/prototype_post_grad_reduction_elim.py` (392 lines) | Prototype FX pass - useful reference but may not be production-ready |
| `scripts/bench_deferred_assertions.py` (127 lines) | Benchmarks a specific optimization - useful but narrow |
| `scripts/bench_online_softmax.py` (490 lines) | Benchmarks online softmax - useful reference |
| `scripts/bench_recompute_fusion_configs.py` (338 lines) | Benchmarks recompute configs - useful reference |
| `scripts/test_pointwise_cat_ab.py` (304 lines) | Tests specific fix |
| `scripts/test_pointwise_cat_fix.py` (141 lines) | Tests specific fix |
| `scripts/test_post_grad_reduction_elim.py` (34 lines) | Minimal test |
| `scripts/test_scatter_reduce_detection.py` (256 lines) | Tests scatter reduce detection |

### Results That May Be Authoritative

| File | Issue |
|------|-------|
| `results/full_corpus_sweep_b200_2026-06-09.json` (16K lines) | Latest full-corpus sweep - could be the canonical B200 baseline |
| `results/full_corpus_sweep_b200_2026-06-09_summary.json` | Summary of above |
| `results/full_graph_sweep_b200_2026-06-09.json` (28K lines) | Full graph sweep - authoritative but large |
| `results/full_graph_genai_sweep_b200_2026-06-09.json` | GenAI subset |
| `results/full_graph_vllm_sweep_b200_2026-06-09.json` | vLLM subset |
| `results/oom_rerun_singles_b200_2026-06-09.json` | OOM rerun results |
| `benchmarks/full_graph_validation_b200.json` (14K lines) | Full graph validation data |

### Investigation Results That Have Reference Value

| File | Issue |
|------|-------|
| `investigation_results/SOL_GAP_SUMMARY.md` | Good summary but may duplicate kernel_family_classification |
| `investigation_results/top_gap_analysis.md` | Useful analysis |
| `investigation_results/non_fusible_ops_benchmark.md` | Useful for understanding model floors |
| `investigation_results/non_fusible_ops_accounting.csv` | Accounting data |
| `investigation_results/non_fusible_ops_measured.json` | Measured data |
| `investigation_results/online_softmax_analysis.md` | Pattern analysis |
| `investigation_results/multi_output_reductions_analysis.md` | Pattern analysis |
| `investigation_results/multi_output_implementation.md` | Implementation notes |
| `investigation_results/reduction_chaining_prototype.md` | Prototype doc |
| `investigation_results/scatter_reduce_prototype.md` | Prototype doc |
| `investigation_results/structured_scatter_reduce_analysis.md` | Analysis |
| `investigation_results/softmax_codegen_gap.md` | Gap analysis |
| `investigation_results/model_level_projection.md` | Model-level impact projection |
| `investigation_results/model_level_projection.csv` | Data backing the projection |
| `investigation_results/non_fusible_ops_benchmark.md` | Non-fusible ops benchmark methodology |

### Inductor Writeups (Family-Level, Not Per-Repro)

| File | Issue |
|------|-------|
| `investigation_results/inductor_writeups/combo_horizontal_tiny_graphlets.md` | Design writeup |
| `investigation_results/inductor_writeups/irregular_gather_reduce.md` | Pattern analysis |
| `investigation_results/inductor_writeups/layout_stencil_functional_updates.md` | Pattern analysis |
| `investigation_results/inductor_writeups/layout_stencil_impl.md` | Implementation notes |
| `investigation_results/inductor_writeups/multi_output_reductions.md` | Pattern analysis |
| `investigation_results/inductor_writeups/norm_templates_bn_ln_rms.md` | Pattern analysis |
| `investigation_results/inductor_writeups/norm_templates_impl.md` | Implementation notes |
| `investigation_results/inductor_writeups/online_softmax_large_row.md` | Pattern analysis |
| `investigation_results/inductor_writeups/online_softmax_large_row_patch_plan.md` | Patch plan |
| `investigation_results/inductor_writeups/online_softmax_patch_plan.md` | Patch plan |
| `investigation_results/inductor_writeups/softmax_backward_attention.md` | Pattern analysis |
| `investigation_results/inductor_writeups/softmax_backward_attention_impl.md` | Implementation |
| `investigation_results/inductor_writeups/structured_pool_upsample_reduce.md` | Pattern analysis |

These family-level writeups are high-quality design documents. They describe optimization opportunities that apply across hundreds of repros. Worth merging if the intent is to share knowledge.

### CSVs with Measured Data

| File | Issue |
|------|-------|
| `investigation_results/baseline_results.csv` | 1482-row baseline - potentially canonical |
| `investigation_results/interleaved_3config_results.csv` | 4446-row final measurements |
| `investigation_results/sol_gap_candidates.csv` | 1090 candidates above 1.1x |
| `investigation_results/gap_groups.csv` | Grouped by prefix |
| `investigation_results/launch_adjusted_priorities.csv` | Launch-adjusted priority ranking |
| `investigation_results/oracle_vs_compile_tracker.csv` | Oracle measurement results |
| `investigation_results/oracle_floor_tracker.csv` | Floor tracking |
| `investigation_results/measured_oracle_floors.csv` | Measured floors |
| `investigation_results/oracle_validation_results.csv` | Validation output |
| `investigation_results/per_repro_realistic_floors.csv` | Per-repro floor estimates |
| `investigation_results/realistic_floor_summary.csv` | Summary |
| `investigation_results/floor_realism_summary.csv` | Realism summary |
| `investigation_results/benchmark_results.sqlite` | SQLite database |

These contain valuable measurement data but are large and may go stale. If merging, they should go under `results/` or a dated `benchmarks/` directory.

---

## Recommended Merge Strategy

### Phase 1: Merge infrastructure (clean, no conflicts expected)
1. `oracle_harness.py`
2. `full_graph_harness.py` patch (the dtype clamp fix)
3. `scripts/oracle_template.py`
4. `scripts/ORACLE_FORMAT.md`
5. `scripts/oracle_dispatch_design.md`
6. `scripts/validate_oracles.py`
7. `scripts/batch_convert_oracles_to_harness.py`
8. `scripts/compare_sweeps.py`
9. `scripts/export_investigation_tables.py`
10. `scripts/_single_repro_runner.py`
11. `scripts/rerun_oom_singles.py`
12. `scripts/backfill_integer_constraints.py`
13. `scripts/retroactive_bounds_inference.py`
14. `scripts/model_attribution.py`
15. `scripts/model_graph_accounting.py`
16. `scripts/measure_all_oracles.py` (fix hardcoded path first)
17. `scripts/build_gap_tracker.py` (fix hardcoded path first)
18. `scripts/export_realistic_floors.py`
19. `optimal_kernels/` (all 5 files)

### Phase 2: Merge documentation
1. `AGENT_INSTRUCTIONS.md`
2. `INVEST_INSTRUCTIONS.MD`
3. `ORACLE_GAP_CLOSING_PLAYBOOK.md`
4. `investigation_results/kernel_family_classification.md`
5. `investigation_results/kernel_priority_by_model_impact.md`
6. `investigation_results/optimization_status_june2026.md`

### Phase 3: Consider merging (after cleanup)
1. Family-level writeups from `investigation_results/inductor_writeups/` (non-per-repro)
2. Canonical measurement data (baseline_results.csv, interleaved_3config_results.csv)
3. `results/` directory (dated sweep results as canonical baselines)

### DO NOT merge
- Per-repro writeups (1479 files)
- Root-level sweep_*.json files (session artifacts)
- Session queues and trackers
- PROGRESS.md
- Root-level investigation_*.md files
- One-off test scripts (test_tl_trans_perf.py, etc.)
