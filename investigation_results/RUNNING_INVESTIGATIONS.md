# Running SOL-Gap Investigations

## Current state

- Results database created under `investigation_results/`.
- Classification completed for all 1090 current candidates where `best_compile_us / memcopy_sol_us > 1.1` using four parallel slices.
- Optimization tracker initialized at `investigation_results/inductor_optimization_tracker.csv`.
- CSV queue initialized at `investigation_results/RUNNING_INVESTIGATIONS.csv`.
- Launch-adjusted priority table initialized at `investigation_results/launch_adjusted_priorities.csv` to avoid over-prioritizing tiny many-launch fusions.
- Full interleaved 3-config sweep is complete: 1482/1482 repros, 0 failures, all 1090 SOL-gap candidates covered.

## Subagent status

- All family investigation subagents are complete; results are consolidated below.

## Final 3-config coverage

- `sweep_3config_interleaved.json` metadata reports `ok=1482`, `total=1482`, `failed=0`.
- `investigation_results/interleaved_3config_results.csv` has 4446 rows: 1482 repros × 3 configs.
- All 1090 raw `best_compile / SOL > 1.1x` candidates are covered by interleaved timings.
- Winners by repro: `combo_looped` 645, `combo_persistent` 569, `default` 268.

## Investigation queue

1. `structured_pool_upsample_backward_reduce` — top priority because it owns the largest gaps, including `sum_sum_sum_45f02142ecfd` at 61.9x SOL.
2. `multi_output_reduction_templates` — broadest high-impact reduction work, spanning `sum_sum*`, `sum_sum_sum*`, and norm backward statistics.
3. `online_softmax_cross_entropy` — large masked attention and vocab reductions where online row kernels should beat generic lowering.
4. `norm_template_canonicalization` — high-count BN/LN/RMSNorm family, especially long tail.
5. `layout_indexing_stencil_fusion` — pointwise/stencil/layout cases where eliminating materialization matters more than generic fusion.
6. `tiny_graphlet_horizontal_fusion` — launch overhead tail cleanup.
7. `irregular_gather_reduce` — investigate only after reusable gather/sink patterns are confirmed.

## Refresh command

```bash
python scripts/export_investigation_tables.py
```

## Suggested per-family workflow

- Select 3-5 representatives from `sol_gap_candidates.csv`.
- Cross-check `launch_adjusted_priorities.csv` to separate true excess time from launch-only ratio artifacts.
- Capture current compile, coordinate-descent, and 3-config timings.
- Write Triton/Helion prototype kernels or series for the family.
- Use `ncu` only after a prototype is functionally correct and clearly faster/slower than compile timing.
- Record best prototype timing, compile comparison, and required Inductor changes in this document and the optimization tracker.

## P0 investigation notes

### `structured_pool_upsample_backward_reduce`

- Status: subagent analysis complete.
- Launch-adjusted result: still P0; representative 3us launch-adjusted ranks are 1, 2, 8, 9, 18, and 28.
- Common subgraph: `full(0) + index_put/scatter_add(accumulate=True) + where + sum([0,2,3])`, with dense scatter output only feeding reductions.
- Fast kernel target: compute channel reductions directly from source grad plus structured offsets/indices; avoid materializing dense scattered tensors such as `[8,64,320,479]`, `[32768,12321]`, or `[65536,3025]`.
- Inductor action: add a semantic rewrite/template for maxpool-offset and bilinear-upsample backward reductions, with multi-output channel reductions sharing one pass.
- First reps: `sum_sum_sum_45f02142ecfd`, `sum_sum_sum_f90d684d32cb`, `sum_sum_sum_dadf6aa035dd`, `sum_18262b26934c`, `sum_3ee4028cab37`, `sum_sum_8bcd6e12dcd4`.

### `online_softmax_cross_entropy`

- Status: subagent analysis complete.
- Launch-adjusted result: split P0s are `sum_sum_sum_afd118ca907d` and `amax_sum_sum_dc96a87ba8db`; Longformer raw gaps drop after launch adjustment but remain useful for mask/window fusion.
- Motifs: pure online softmax, CE/log-softmax+NLL, Longformer sliding-window masked softmax, and softmax-backward/attention-backward fragments.
- Fast kernel target: online row kernels for large vocab/softmax rows; direct CE without full log-softmax materialization; fused softmax-backward pattern using saved or recomputed `m/l`.
- Inductor action: pattern-match `amax -> sub -> exp -> sum -> div`, CE gather/masked-mean, and `p = exp(x-m)/l; p*(g-sum(g*p))` softmax backward; favor online for large no-dropout `rnumel >= 4096`.
- First reps: `sum_sum_sum_afd118ca907d`, `amax_sum_sum_dc96a87ba8db`, `amax_sum_sum_9e768dff4978`, `amax_sum_87e1fb077f24`, `amax_sum_9940b361e5b4`.

### `multi_output_reduction_templates`

- Status: subagent analysis complete.
- Launch-adjusted result: best reps remain meaningful by excess time; `sum_sum_sum_70d71fcb0d68` rank 26, `sum_sum_sum_95dac16d4328` rank 40, and `var_mean_var_mean_6d7a29cb97f1` rank 52 at 3us launch floor.
- Common subgraph: sibling reductions over same logical input and same or related reduced axes, usually with lightweight `scale/sub/mul` epilogues and sometimes a full-sized output sharing the same source.
- Fast kernel targets: BN-backward one-pass `(sum_x, sum_x_centered_or_grad[, dx])`; Welford/norm-forward `(var, mean)` plus affine/running-stat epilogue; LN-backward row template; staged GRN backward reductions grouped by compatible axes.
- Inductor action: expose same-input sibling reductions to template selection; reward fusing same-base/same-axis reductions when accumulator count is small; penalize rereads and incompatible-axis fusion in scheduler cost model.
- First reps: `sum_sum_sum_70d71fcb0d68`, `sum_sum_sum_95dac16d4328`, `var_mean_var_mean_6d7a29cb97f1`, `sum_sum_ee85624361a0`, `sum_sum_sum_bfac9a5afa42`.
- Relevant hooks: multi-output reduction IR, Welford lowering, generic `sum` lowering, grouped/mixed reduction scheduling, and persistent-reduction heuristics in Inductor.

### `norm_template_canonicalization`

- Status: subagent analysis complete.
- Launch-adjusted result: real P1; `var_mean_765fb8f2c85e` rank 29 with 632us excess, `var_mean_var_mean_var_mean_0407b3e7c77f` rank 93, and `var_mean_598830735cf6` rank 119.
- Common motifs: decomposed BN training forward with `var_mean.correction([0,2,3]) -> rsqrt -> affine -> ReLU -> running-stat copy_`; multi-branch Inception BN+cat; RMSNorm as `pow(x,2) -> mean(-1) -> rsqrt -> scale` after dropout/embedding.
- Fast kernel target: dedicated BN training forward template reducing over `N*H*W` per channel with affine/ReLU epilogue and save/running-stat outputs; row-wise RMSNorm/LN persistent template for hidden-dim norms.
- Inductor action: recover normalized composite semantics in post-grad patterns, bypass brittle generic `var_mean` lowering when a full BN/LN/RMSNorm motif is recognized, and add `TritonTemplate` choices for recognized composites.
- First reps: `var_mean_765fb8f2c85e`, `var_mean_598830735cf6`, `var_mean_var_mean_var_mean_0407b3e7c77f`, then `mean_5c93e9826aa8` and `mean_mean_1b98d81214e6`.

### `layout_indexing_stencil_fusion`

- Status: subagent analysis complete.
- Launch-adjusted result: mixed; `pointwise_531d72f1b34a` is heavily launch-inflated, but `pointwise_70c0751eb408`, `pointwise_a2382a85ee99`, and `pointwise_e26de0a669ae` still have real adjusted excess.
- Common motifs: PyHPC halo/interior f64 stencil windows around `204x204x26` grids, rebuilt through `slice/copy/slice_scatter/select_scatter`; pure large f64 expression kernels with register/CSE pressure; channels-last-ish BN/ReLU/pooling chains; dropout/RNG graphlets.
- Fast path: Inductor canonicalization first, not hand Triton first. Convert chained rectangular functional updates into symbolic offset loads/stores over interior domains, then let fusion/codegen produce one/few kernels.
- Inductor action: pattern-match `slice/select -> copy -> slice_scatter/select_scatter`, avoid full-tensor reconstruction for simple updates, add stencil-domain IR/canonicalization, and improve f64 expression scheduling/CSE.
- First reps: `pointwise_70c0751eb408`, `pointwise_531d72f1b34a`, `pointwise_a2382a85ee99`, `pointwise_e26de0a669ae`, `pointwise_3229f6e30711`.

### `tiny_graphlet_horizontal_fusion`

- Status: subagent analysis complete.
- Launch-adjusted result: keep as gated P2 cleanup only. `pointwise_c14f03aac63b` and `sum_sum_sum_92327e661e73` are raw-gap false positives after launch floor; `sum_sum_sum_3cd8c07ebace` remains a better adjusted-positive tiny/many-kernel representative.
- Triage thresholds: ignore if `launch_adjusted_gap_3p0 <= 1.1` or `excess_us_vs_launch_adjusted_sol_3p0 <= 0`; consider P2 if `n_kernels >= 8`, `best_compile_us <= 750us`, `launch_adjusted_gap_3p0 >= 1.25`, and excess >= 50us; prioritize strongly if excess >= 150us or score >= 200.
- Fast path: improve combo-kernel horizontal partitioning with launch-savings vs in-kernel dispatch overhead, not a standalone new template.
- Inductor action: tune `_default_custom_combo_kernel_horizontal_partition`, `combo_kernel_per_subkernel_blocks`, `benchmark_combo_kernel`, x-dimension ratio guard, and minimum runtime thresholds.
- First reps: adjusted-negative controls `pointwise_c14f03aac63b`, `sum_sum_sum_92327e661e73`; adjusted-positive controls `sum_sum_sum_3cd8c07ebace`, `amax_sum_67d7c2666a5c`, `sum_sum_sum_f48adec0ff36`, `pointwise_531d72f1b34a`.

### `irregular_gather_reduce`

- Status: subagent analysis complete.
- Launch-adjusted result: `sum_adeaebad93f7` remains high by excess time, but memcopy SOL is not a realistic target because irregular gather/sort/cat traffic dominates and byte accounting underestimates source/intermediate effects.
- Common motif: Demucs augmentation with tiny random sort/permutation, channel flip/sign, time-window gather via offsets, group permutation gather, then `sum(dim=1)`; no other exact gather+sort+cat+sum candidate found.
- Fast kernel target: two-stage lowering — keep tiny RNG/sort/index generation separate, then emit one fused gather-reduce kernel computing `out[b,c,t] = sum_s signed input[selected batch, stream, swapped c, offset+t]` over `[64,2,382788]`.
- Inductor action: pattern rewrite for `gather(cat(gather(x,left), gather(x,1-left)), arange+offset) -> gather_from_base`, plus reduction sinking through pure `gather/view/gather` when reducing over stream axis.
- First rep: `sum_adeaebad93f7`; treat as low-medium scope because scan found no broad duplicate family.

## Floor realism

We now have three floor levels, but only the first two are systematic for every repro:

1. `memcopy_sol_us`: available for all 1482 baseline rows; useful as an absolute lower bound, often too optimistic for reductions, scatters, gather/sort, and math-heavy softmax.
2. `launch_adjusted_sol_us_{1,3,5}`: available for all rows via `scripts/export_investigation_tables.py`; better for triaging tiny many-launch false positives.
3. Semantic/prototype floors: available only by family analysis today; these are needed before claiming realistic speedup potential for scatter-reduce, BN/LN/RMSNorm, online softmax/CE, stencil/layout, and irregular gather.

See `investigation_results/floor_realism_summary.csv` for the current confidence table. The practical answer is: we have defensible triage floors for all candidates, but realistic implementation floors only for families where we already have heuristics/prototype directions, especially online softmax. The biggest missing realistic floors are structured scatter-reduce and norm templates.

## Floor prioritization decision

I will not try to build realistic floors for every candidate before implementation. That would slow down the high-value work and many candidates already have enough launch-adjusted evidence for triage.

Prioritize realistic floors only when they could change implementation order or expected ROI:

1. Build/measure prototype floors for P0 scatter-reduce, multi-output reductions, and softmax-backward.
2. Use existing online-softmax heuristic data as a sufficient floor for large vocab/softmax forward unless a specific repro disagrees.
3. Build P1 floors for BN training forward and PyHPC stencil canonicalization before implementing broad Inductor changes.
4. Defer P2 floors unless a P2 item becomes an implementation target.

See `investigation_results/realistic_floor_worklist.csv` for the concrete ordered worklist.

## Realistic floor clarification

A truly realistic floor means a measured oracle/prototype kernel (or kernel series) for the repro family, not just `memcopy_sol_us` or a heuristic multiplier. The heuristic files are still useful for triage, but they are not proof of achievable performance.

- Heuristic estimates: `investigation_results/per_repro_realistic_floors.csv` and `investigation_results/realistic_floor_summary.csv`.
- Actual measured-floor tracker: `investigation_results/oracle_floor_tracker.csv`.
- Current measured oracle floors: none yet recorded in this pass.
- First recommended oracle target: likely `amax_sum_sum_dc96a87ba8db` or the simpler maxpool-offset direct reduce (`sum_18262b26934c`), pending subagent feasibility results.

## Oracle kernel placement rule

Implemented oracle/prototype kernels should live inside the owning canonical repro directory, not only under `optimal_kernels/`. Example:

- `repros/canonical/amax_sum_sum_dc96a87ba8db/oracle_softmax_sum.py`

This keeps each measured floor adjacent to the repro inputs/shapes and makes per-repro correctness/benchmark commands discoverable.

## Inductor implementation starting point

After canonical oracle scaffolds and `investigation_results/oracle_floor_tracker.csv` are updated, the next phase is implementing Inductor optimizations. Existing local work to reuse:

- Stash: `stash@{0}` labeled `all local optimizations`.
- Included work: scalar accumulators, tiling scores, `R0_BLOCK` expansion, persistent threshold, `num_warps=2`, one-hot→gather, and online softmax fixes.

Planned order after floor/oracle pass:

1. Re-apply/review stash changes relevant to `online_softmax_cross_entropy` first, because the oracle scaffold exists for `amax_sum_sum_dc96a87ba8db`.
2. Then evaluate reduction config / multi-output reduction changes against the P0/P1 oracle targets.
3. Keep oracle measurements in canonical repro directories and append measured rows to `investigation_results/measured_oracle_floors.csv`.

## Canonical oracle scaffold batch

Canonical-local oracle scaffolds are now in place for the main priority families. They are `implemented_unmeasured` because this environment could not initialize CUDA; syntax checks and CPU/no-CUDA smoke tests were used where possible.

- `repros/canonical/amax_sum_sum_dc96a87ba8db/oracle_softmax_sum.py`
- `repros/canonical/sum_18262b26934c/oracle_maxpool_direct_reduce.py`
- `repros/canonical/sum_sum_sum_70d71fcb0d68/oracle_multi_output_reduction.py`
- `repros/canonical/var_mean_765fb8f2c85e/oracle_bn_training_forward.py`
- `repros/canonical/pointwise_70c0751eb408/oracle_stencil_canonicalized.py`
- `repros/canonical/sum_adeaebad93f7/oracle_fused_gather_reduce.py`

`investigation_results/oracle_floor_tracker.csv` is updated with scaffold paths and benchmark status. Next, run these scripts on the CUDA/B200 host to append measured rows to `investigation_results/measured_oracle_floors.csv`, then start Inductor optimization from `stash@{0}`.

## Inductor optimization objective

For the implementation phase, optimize for best measured performance under coordinate-descent/autotuning, not for default heuristic quality.

Target comparison mode:

- Enable `coordinate_descent_tuning=True`.
- Compare forced combo variants rather than relying on heuristics:
  - non-looped / persistent-style config: `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2`
  - looped config: `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`
- Treat `benchmark_combo_kernel=True` and broader autotuning as acceptable compile-time cost.
- Prioritize best runtime among tuned variants and oracle-informed implementations, not static heuristic selection.

This means implementation work can add/search more candidates, larger `R0_BLOCK`, alternate `num_warps`, persistent thresholds, looped/non-looped variants, and semantic templates even if default heuristics would not choose them without benchmarking.
