# Kernel Family Classification for `best_compile / SOL > 1.1x`

Generated from `sweep_pr184905_baseline.json` plus parallel classification of the 1090 repros where the faster of default compile and coordinate-descent compile is still more than 1.1x over memcopy SOL.

Artifacts:
- `investigation_results/sol_gap_candidates.csv`: every candidate repro/case above 1.1x.
- `investigation_results/baseline_results.csv`: all 1482 baseline rows.
- `investigation_results/interleaved_3config_results.csv`: final 3-config interleaved results for all 1482 repros.
- `investigation_results/gap_groups.csv`: grouped by repro name prefix.
- `investigation_results/benchmark_results.sqlite`: SQLite copy of all tables.

## Coverage snapshot

- Baseline rows: 1482
- Candidate rows above 1.1x: 1090
- Candidate rows above 2.0x: 544
- Candidate rows above 5.0x: 67
- Candidate pattern-prefix groups: 29
- Final interleaved coverage: 1482 repros / 4446 config rows
- Final 3-config winners among covered repros: `combo_looped` 645, `combo_persistent` 569, `default` 268

## Launch-adjusted prioritization

Raw `best_compile / memcopy_SOL` over-prioritizes some tiny, many-launch fusions. The exporter now adds launch-adjusted floors for 1us, 3us, and 5us per compiled kernel:

- `launch_adjusted_sol_us_L = memcopy_sol_us + L_us * n_kernels`
- `launch_adjusted_gap_L = best_compile_us / launch_adjusted_sol_us_L`
- `launch_adjusted_excess_us_L = best_compile_us - launch_adjusted_sol_us_L`
- `priority_score_3us = max(0, launch_adjusted_excess_us_3us) * log2(raw_sol_gap)`

This creates `investigation_results/launch_adjusted_priorities.csv`. The highest-priority cases remain real bottlenecks, but launch artifacts fall sharply: `pointwise_531d72f1b34a` moves from raw rank 6 to much lower priority because it has 116 kernels and only 37.7us bare SOL, while `pointwise_70c0751eb408` similarly drops due to 67 kernels. The top 3us launch-adjusted priorities are now `sum_sum_sum_45f02142ecfd`, `sum_sum_sum_f90d684d32cb`, `sum_sum_sum_afd118ca907d`, `sum_sum_sum_7b24a3457260`, and `sum_adeaebad93f7`. The completed interleaved sweep covers all 1090 raw SOL-gap candidates, so these priorities can now be compared against final `default`, `combo_persistent`, and `combo_looped` timings.

## Aggregate prefix groups

| Prefix | Count >1.1x | Max gap | Geomean gap | Representative repros |
|---|---:|---:|---:|---|
| `var_mean` | 224 | 7.52x | 1.67x | `var_mean_765fb8f2c85e`, `var_mean_f0d7c08a0622`, `var_mean_598830735cf6` |
| `sum_sum_sum` | 205 | 61.88x | 2.84x | `sum_sum_sum_45f02142ecfd`, `sum_sum_sum_f90d684d32cb`, `sum_sum_sum_dadf6aa035dd` |
| `pointwise` | 178 | 14.56x | 1.92x | `pointwise_531d72f1b34a`, `pointwise_70c0751eb408`, `pointwise_a14dcfc06344` |
| `sum_sum` | 135 | 11.15x | 2.27x | `sum_sum_8bcd6e12dcd4`, `sum_sum_3219a09ab96a`, `sum_sum_6a14a9c9ba88` |
| `sum` | 109 | 15.16x | 1.96x | `sum_adeaebad93f7`, `sum_18262b26934c`, `sum_3ee4028cab37` |
| `mean` | 51 | 4.45x | 1.85x | `mean_5c93e9826aa8`, `mean_d0fc206717a8`, `mean_f7170c220032` |
| `amax_sum` | 38 | 10.81x | 3.09x | `amax_sum_9940b361e5b4`, `amax_sum_4c524f75213e`, `amax_sum_67d7c2666a5c` |
| `amax_sum_sum` | 30 | 5.22x | 2.79x | `amax_sum_sum_dc96a87ba8db`, `amax_sum_sum_9e768dff4978`, `amax_sum_sum_1bad0f362efd` |

## Priority families

### P0: Structured scatter/upsample/pool backward plus channel reductions

- Representative repros: `sum_sum_sum_45f02142ecfd` (61.9x), `sum_sum_sum_f90d684d32cb` (36.3x), `sum_sum_sum_dadf6aa035dd` (21.6x), `sum_18262b26934c`, `sum_3ee4028cab37`, `sum_sum_8bcd6e12dcd4`.
- Common ops/shapes: `constant_pad_nd`, bilinear-ish `index_put(accumulate=True)`, `_low_memory_max_pool_offsets_to_indices`, `scatter_add`, `where`, and one or more `sum([0,2,3])`; CNN/UNet shapes from `f32[128,64,112,112]` to `f32[8,128,640,959]`.
- Why current compile is far from SOL: Inductor materializes the full scattered/upsampled gradient, then rereads it for channel reductions; atomics and offset conversion block coalescing; sibling reductions duplicate input reads or become large fused kernels with poor locality/register pressure.
- Fastest-kernel direction: custom Triton/Helion series for maxpool/upsample backward directly producing channel stats, avoiding full dense scatter materialization when only reductions are consumed.
- Inductor optimization: structured rewrite for maxpool/upsample backward + reductions; direct-reduce from pooled grad and offsets; multi-output channel reductions sharing one pass.

### P0: Multi-output dense reductions and norm backward statistics

- Representative repros: `sum_sum_sum_95dac16d4328`, `sum_sum_sum_70d71fcb0d68`, `sum_sum_sum_431633879271`, `sum_sum_sum_bfac9a5afa42`, `sum_sum_ee85624361a0`, `var_mean_var_mean_6d7a29cb97f1`.
- Common ops/shapes: repeated `aten.sum.dim_IntList`, `aten.var_mean.correction`, `mul/sub/rsqrt/copy_`, reductions over `[0,2,3]`, `[0,1]`, `[2]`, and mixed keepdim outputs; BN/LN/backward shapes such as `[128,320,56,56]`, `[512,40,28,28]`, `[8,512,80,119]`, `[128,128,768]`.
- Why current compile is far from SOL: same source tensor is read multiple times; generic fused reductions combine incompatible sub-reductions; per-channel stats and affine/running-stat updates are split into multiple kernels or produce high register pressure.
- Fastest-kernel direction: one-pass multi-output Welford/sum kernels for norm backward; tuned dense reductions split by compatible axes and reduction sizes.
- Inductor optimization: multi-output reduction templates for repeated same-input sums, norm-backward statistic clustering, and cost model to split incompatible reductions instead of blindly comboing.

### P0: Online softmax, attention, and cross-entropy/vocab reductions

- Representative repros: `amax_sum_9940b361e5b4`, `amax_sum_4c524f75213e`, `amax_sum_87e1fb077f24`, `amax_sum_amax_2a81770def44`, `amax_sum_sum_dc96a87ba8db`, `amax_sum_sum_9e768dff4978`, `sum_sum_sum_afd118ca907d`.
- Common ops/shapes: `amax`, `exp`, `sum`, `div`, `log`, `gather`, `where`, `ne`, `topk`, `view/permute/slice`; Longformer blocks like `[8,12,1024,513]`, attention matrices `[64|96,1024,1024]`, vocab rows `[2048,50272..151936]` and `[8192,30522/50265]`.
- Why current compile is far from SOL: max/sum/normalize/log/gather are separate passes; masks and slice/scatter windows create layout movement; generic lowering can materialize log-softmax-sized intermediates for huge vocab rows.
- Fastest-kernel direction: Triton online row reductions, FlashAttention-style masked softmax, and cross-entropy kernels that compute max/sum/gathered loss in one row pass without materializing full outputs.
- Inductor optimization: pattern-match `amax + exp + sum + div + where`, `log_softmax + nll_loss`/cross-entropy, and masked softmax backward into persistent online templates.

### P1: BatchNorm/LayerNorm/RMSNorm forward and training chains

- Representative repros: `var_mean_598830735cf6`, `var_mean_765fb8f2c85e`, `var_mean_f0d7c08a0622`, `var_mean_var_mean_var_mean_0407b3e7c77f`, `mean_5c93e9826aa8`, `mean_d0fc206717a8`, `mean_mean_1b98d81214e6`.
- Common ops/shapes: `var_mean.correction`, `mean.dim`, `var.correction`, `rsqrt`, fp32 accumulation, casts, affine epilogues, running-stat `copy_`; NCHW/channel-last shapes with small C and large spatial, plus row norms with dim 512/768/1024.
- Why current compile is far from SOL: generic two-stat reductions and side-output updates reread data; keepdim/correction variants block canonicalization; small row reductions are launch/iterator overhead dominated.
- Fastest-kernel direction: Helion/Triton norm kernels specialized by row/channel shape with affine epilogue and optional running-stat updates.
- Inductor optimization: canonical LN/RMSNorm/BN templates; strain-aware NCHW/NHWC reductions; warp-specialized small reductions; affine/running-stat epilogue fusion.

### P1: Pointwise/layout/indexing and stencil update graphlets

- Representative repros: `pointwise_531d72f1b34a`, `pointwise_70c0751eb408`, `pointwise_a2382a85ee99`, `pointwise_3229f6e30711`, `pointwise_437415a3398d`, `pointwise_71e3a6c09140`, `pointwise_e26de0a669ae`.
- Common ops/shapes: `slice/select/select_scatter/slice_scatter`, `permute`, `clone`, `cat`, `expand`, `gather`, `index_put`, dropout RNG, BN affine/ReLU, pooling offsets; f64 grid stencils like `[204,204,26,3]`, Reformer/Longformer layouts, and CNN tensors.
- Why current compile is far from SOL: aliasing and scatter updates force kernel explosion; non-contiguous indexing reduces coalescing; pool/cat/RNG boundaries inhibit fusion; many tiny graphlets pay launch overhead.
- Fastest-kernel direction: custom/Helion stencil update series, gather/scatter-aware vectorized Triton kernels, and better layout elimination rather than more generic fusion.
- Inductor optimization: fusion through functionalized slice/scatter SSA updates, stencil recognition, layout-aware cost model for permute/clone/cat, RNG+dropout+residual fusion, and launch-overhead-aware horizontal fusion for tiny graphlets.

### P2: Irregular gather/sort reductions

- Representative repro: `sum_adeaebad93f7` (15.2x).
- Common ops/shapes: `sort`, repeated `gather`, RNG seed/lookups, `cat`, final `sum` over `f32[64,5,2,426888]`.
- Why current compile is far from SOL: irregular gather/sort destroys coalescing and inflates reads before a simple reduction; memcopy SOL is a weak lower bound for this pattern.
- Fastest-kernel direction: custom Triton only if indices are predictable or reusable; otherwise treat as less immediately actionable than structured reductions.
- Inductor optimization: coalesce repeated gathers and sink reductions through gather/cat when algebraically safe.

## Next classification steps

- Refresh tables with `python scripts/export_investigation_tables.py` as `sweep_3config_interleaved.json` continues to grow.
- Assign optimization-owner subagents by priority family, starting with structured scatter+reduction, multi-output reductions, and online softmax/cross-entropy.
- For each family, pick 3-5 representative repros from `sol_gap_candidates.csv`, hand-write or autotune Triton/Helion kernels, compare against compile and current 3-config timings, then record achieved speedups in a per-family table.
