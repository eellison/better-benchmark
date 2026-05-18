# Benchmark Infrastructure Roadmap

## Track 1: Core Utilities (benchmark infra quality)

### Done
- bench_parallel.py with persistent workers, GPU locks, respawn on CUDA error
- capture_hook.py with shape lifting for reshape/view/expand
- repro_prelude.py for clean imports
- Per-model JSON manifests (git-friendly)
- Compare workflow (--tag baseline, --tag fix, --compare)
- CUDA graph + do_bench timing
- Memcopy SOL per-shape baseline

### Next
- [x] ~~Better index bounds in capture_hook~~ (DONE: infer from scatter/gather/embedding target dims)
- [ ] GPU lock hook inside inductor autotuning: add `with gpu_benchmark_lock():` around autotune trials in `triton_heuristics.py`. This lets multiple processes compile in parallel (CPU-bound graph lowering) while serializing only the GPU autotune trials. Combined with multi-process compile, could reduce sweep time from ~7s/repro to ~2-3s/repro (overlap CPU compilation across processes, only serialize GPU timing). Key insight: compilation is CPU until autotuning, then brief GPU, then CPU again for codegen.
- [ ] Fix shapes.json to include S() shape param entries (currently --all-shapes breaks with lifted repros)
- [ ] Compute-intensity metric: tag kernels as memory-bound vs compute-bound (FLOPs/byte ratio) so SOL gap is only reported when meaningful
- [ ] Incremental results file (JSONL) during sweep — survive interrupts
- [ ] Regression alerting: after a sweep, auto-flag any pattern that regressed >5% vs previous tag
- [ ] Profile mode: `python repro.py --profile` emits torch.profiler trace or Triton TTGIR for deep investigation

### Recently Completed
- [x] `bench_parallel.py --benchmark-set benchmarks/v1.json` (runs frozen versioned set, 3557 points, ~15 min)
- [x] Smart shape sampling (`scripts/select_benchmark_set.py` — quantile-grouped pointwise + all reduction shapes)
- [x] Lift reshape/view/expand dims to _shape_param (1069 repros converted)
- [x] Fix missing output dtype cast (52 repros regenerated)
- [x] Per-model JSON manifests (263 model files, git-friendly)
- [x] repro_prelude.py (all 1521 repros use it)
- [x] Handle symbolic shapes / SymInt (resolve to concrete hint, emit scalar dims)
- [x] `--all-shapes` in repro_harness (works per-repro)

### Polish
- [ ] Fix ndim-mismatch edge case in pattern hash (include input ranks if needed)
- [x] ~~Handle symbolic shapes in make_inputs~~ (DONE: resolve SymInt to concrete hint, emit scalar dims)
- [ ] Faster compilation during capture (disable fusion passes that don't affect graph structure)

---

## Track 2: Expanded Repros (coverage)

### Done
- 1,521 canonical repros from 438 model sources
- HF: 33 models (inference + training)
- timm: 18 models (channels-last, inference + training)
- torchvision: 10 models (inference + training)
- vLLM: 9 model configs (Qwen3, DeepSeek-V3, Mistral, OPT, GPT-OSS)
- GenAI: 8 kernel types at varied shapes (decode through long-context)
- tritonbench: 12 operator categories with reference impls

### Next
- [ ] torchbench: install full repo, capture 58 CI models (especially conv: YOLO, maskrcnn, stable_diffusion_unet, demucs)
- [ ] Conv models with postfusion capture (capture_hook_postfusion.py) for real strides after layout opt
- [ ] More decode shapes: batch=64/128 with seq=1 (production LLM serving regime)
- [ ] Dynamic shape captures: same model at multiple batch sizes to test dynamic shape handling
- [ ] Backward for ALL timm models (currently only 5 have training captures)
- [ ] Quantized models (int8, fp8) — different kernel patterns from float
- [ ] Multi-GPU patterns (tensor parallel, pipeline parallel) — if applicable to kernel level
- [ ] Edge cases: very large (>32GB working set) and very small (<1KB) tensors

### Reference implementations
- [ ] Extract more tritonbench kernels as impls/ (currently have softmax, layer_norm, rms_norm)
- [ ] Add Liger kernel implementations as impls/ (liger-kernel has optimized cross-entropy, RMSNorm, etc.)
- [ ] Add FlashAttention-style decode kernel as impl (our GQA decode is 1.5x off FA3)
- [ ] Helion kernels once available

---

## Track 3: CI / User Integration

### Vision
A developer making an inductor change can:
1. Run `python scripts/bench_parallel.py repros/canonical/ --tag my_fix` (~12 min on 2 GPUs)
2. See instantly: "142 improved, 3 regressed, geometric mean 1.02x speedup"
3. Attach the summary to their PR as evidence

### Next
- [ ] `scripts/bench_report.py` — generates a markdown summary from two tagged runs (for PR description)
- [ ] Model-level aggregation: "BERT training: 3.2% faster overall (5 kernels improved)"
- [ ] Integration with PyTorch CI: run our sweep as a post-commit check (needs Docker image with repros)
- [ ] `scripts/bench_bisect.py` — binary search across git commits to find when a regression was introduced
- [ ] Web dashboard: static HTML showing gap distribution, top regressions, historical trends
- [ ] GitHub Action: on PR, run sweep on changed inductor files, post comment with results
- [ ] Nightly tracking: run sweep every night on main, alert on regressions >3%
- [ ] Per-model regression detection: "this PR makes BERT 5% slower but GPT-2 10% faster — is that okay?"

### Documentation
- [ ] User guide: "How to benchmark your inductor change in 5 minutes"
- [ ] Contributing guide: "How to add a new model to the benchmark set"
- [ ] Interpreting results: "What does a 1.3x gap mean and when should you care?"

---

## Track 4: Inductor Performance Fixes

### Verified fixes (ready to implement/land)

| # | Fix | Speedup | Effort | Files |
|---|-----|---------|--------|-------|
| 1 | Enable combo_kernels by default | 3.0x | Config flip + validation | config.py |
| 2 | ND grid codegen for transpose/NHWC | 3.0-3.4x | Codegen change | simd.py, tiling_utils.py |
| 3 | Relax MixOrderReduction threshold | 2.0x | Threshold change | scheduler.py:326-332 |
| 4 | Rotary embed: eliminate cat | 2.04x | Pattern match pass | fx_passes/ |
| 5 | Cross-entropy: fuse gather into softmax | 1.69x | Graph pass | fx_passes/ |
| 6 | Raise persistent RBLOCK to 4096 on B200 | 1.4-1.6x | Threshold change | triton_heuristics.py:3514 |
| 7 | num_warps=2 for persistent INNER | 1.28x | 1-line change | triton_heuristics.py:4114 |
| 8 | Fuse permute+sum (read once) | 2.15-2.69x | Scheduler fusion | scheduler.py |

### Investigation-informed (need more design)

| # | Fix | Potential | Complexity | Notes |
|---|-----|-----------|-----------|-------|
| 9 | Batched weight-grad reduction kernel | 6.5x | High | New kernel type for scheduler |
| 10 | ConcatKernel virtual views | 1.5-2x | High | Structural change to ir.py |
| 11 | Push reduction through cat (orthogonal axes) | 2x | Medium | Graph rewrite pass |
| 12 | Horizontal fusion for same-shape reductions | 1.5x | Medium | Relax can_fuse threshold |
| 13 | Non-power-of-2 RBLOCK configs | 1.3x | Low | Add configs to autotuner |
| 14 | Duplicate scatter buffer merging | 1.5x | Medium | Graph pass for embedding bwd |
| 15 | SwiGLU decode: better XBLOCK for small tensors | 1.44x | Low | Heuristic tuning |
| 16 | Fast math for online_softmax_combine | 1.1x | Low | Use tl.math.exp |

### Validation workflow for each fix
1. Identify affected repros (from triage_all_v2.json)
2. Run sweep with --tag before
3. Apply fix
4. Run sweep with --tag after
5. `bench_parallel.py --compare before after` → confirm improvement, no regressions
6. Write PR with benchmark evidence

---

## Priority Order

**Immediate high-impact (this week):**
1. Land combo_kernels fix (3x on 60% of severe gaps, config change)
2. Land num_warps/RBLOCK fixes (1.3-1.6x, trivial changes)
3. Run full --all-shapes sweep to establish per-shape baseline

**Short-term (next 2 weeks):**
4. ND grid codegen for transpose (3.4x, high-frequency pattern)
5. MixOrderReduction threshold (2.0x, every norm backward)
6. Rotary embed pattern match (2.04x, every LLM)
7. Set up nightly regression tracking

**Medium-term:**
8. Cross-entropy fuse gather (1.69x, all LLM training)
9. Batched weight-grad kernel (6.5x, needs new scheduler concept)
10. ConcatKernel virtual views (general cat barrier fix)
11. torchbench expansion + postfusion captures
12. CI integration (GitHub Action, dashboard)
