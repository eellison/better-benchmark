# Can a vertical MM region be one kernel? (Helion study)

The `vertical` category of `mm_repro/` (and the MM chains of the previous
largest-window corpus) is meant to collect regions that ONE kernel can run
without recomputation. This study writes such kernels by hand in Helion,
checks that they compile, run correctly and launch exactly one kernel, and
measures when fusion is actually faster. The results are used to judge the
extraction heuristics.

## Setup

- H100, Helion `13b35b48`, Triton 3.9, `autotune_effort="quick"`,
  `static_shapes=True`; the first shape point of each repro.
- **fused**: one `@helion.kernel` with one top-level loop, no `hl.barrier`,
  intermediates kept on chip, no recomputation. The CUDA kernels it launches
  are counted with the profiler: always exactly 1.
- **unfused**: the same region written as one Helion kernel per MM stage,
  with all pointwise/view work folded into the neighboring stage kernel and
  consecutive non-MM reductions over the same rows sharing a kernel. This is
  the best case for separate kernels, so fused/unfused isolates the effect of
  the stage boundaries.
- **inductor**: `torch.compile` defaults (cuBLAS MMs), for reference.
- Timing: the repository's method (3 warmups, CUDA graph, `do_bench` min).
  Correctness: against the repro's own eager `Repro`, bf16-level tolerance.

Files:

| | |
| --- | --- |
| `benchmarks/vertical_helion/new/<repro>/helion_vertical.py` | fused + unfused kernels for vertical repros of the pattern corpus, with the repro they were written for; `scripts/install_vertical_helion.py` copies them into `mm_repro/canonical/vertical/` and writes `benchmarks/vertical_helion/STATUS.md` |
| `benchmarks/vertical_helion/old/<repro>/helion_vertical.py` | same for regions of the previous largest-window corpus |
| `.../helion_vertical_result.json` | checks, kernel counts, timings |
| `scripts/vertical_fusion_bench.py` | harness (`python scripts/with_gpu_lock.py --gpu N -- python <kernel file>`) |
| `scripts/vertical_fusion_summary.py`, `scripts/vertical_fusion_features.py` | the tables below |

"new" below is the pattern extractor as it was during the study (R, F <= 192,
124 vertical repros); "old" is the previous largest-window partitioner and its
corpus (first MM any K, chained K <= 128, reductions <= 128), which this
change removes. 21 new vertical repros (every distinct chain type, plus the
most frequent families) and 5 old-corpus regions were written.

## Results

| corpus | repro | stages (new-spec R/F) | fused us | unfused us (kernels) | inductor us (kernels) | fused/unfused |
| --- | --- | --- | --- | --- | --- | --- |
| new | `addmm_bmm_cd12c211d331` q-proj + attention | addmm -> bmm -> any -> amax -> sum -> bmm | 34.4 | 70.0 (4) | 81.3 (6) | **2.03** |
| new | `addmm_6f811013a726` tiny tanh MLP | addmm -> addmm -> addmm | 4.8 | 7.5 (3) | 13.4 (6) | **1.56** |
| new | `bmm_0f0f8a25f92c` attention fwd | bmm -> any -> amax -> sum -> bmm | 30.7 | 45.3 (3) | 52.2 (4) | **1.48** |
| new | `addmm_798c593e4ee0` tiny BN -> linear -> log_softmax | var_mean -> addmm -> amax -> sum | 5.4 | 7.5 (3) | 10.5 (4) | **1.38** |
| new | `bmm_e90e4276642b` softmax bwd -> dq, dk (4096 slices) | sum -> bmm -> bmm | 81.4 | 112.2 (3) | 149.9 (4) | **1.38** |
| new | `mm_179800676c71` tiny MLP bwd | mm -> mm -> sum | 4.5 | 5.9 (2) | 10.6 (4) | **1.29** |
| new | `addmm_bmm_86f50de0a77b` k-proj -> scores | addmm -> bmm | 46.9 | 57.7 (2) | 91.9 (4) | **1.23** |
| new | `bmm_ac33030a7533` attention fwd | bmm -> amax -> sum -> bmm | 71.1 | 86.8 (3) | 117.7 (4) | **1.22** |
| new | `bmm_9e3be953821a` softmax bwd -> dq, dk (512 slices) | sum -> bmm -> bmm | 69.3 | 83.8 (3) | 104.8 (5) | **1.21** |
| new | `bmm_a7f1e10d4357` softmax bwd -> dq, dk (192 slices) | sum -> bmm -> bmm | 33.8 | 40.3 (3) | 46.6 (5) | **1.19** |
| new | `addmm_41b6b1a7d236` tiny pool -> linear | mean -> addmm | 5.6 | 6.6 (2) | 11.9 (4) | **1.19** |
| new | `bmm_mm_52f22734ce29` ctx -> out-proj -> FFN | bmm -> mm -> mm | 59.6 | 70.2 (3) | 78.7 (5) | **1.18** |
| new | `addmm_bmm_478c63b3f825` softmax -> PV -> proj -> FFN | ... -> bmm -> addmm -> addmm | 77.0 | 85.7 (3) | 124.5 (7) | **1.11** |
| new | `bmm_01f3eb169a50` softmax -> PV | amax -> sum -> bmm | 7.1 | 7.9 (2) | 10.8 (3) | **1.11** |
| new | `bmm_1d7e3367c27c` softmax(9) -> N=1 matvec | amax -> sum -> bmm | 84.4 | 92.2 (2) | 154.9 (3) | 1.09 |
| new | `bmm_mm_3759ef405a77` PV -> weight grad | bmm -> mm | 33.8 | 35.3 (2) | 70.1 (4) | 1.05 |
| new | `mm_83df85014e51` norm bwd -> mm | sum -> sum -> mm | 287.1 | 284.2 (2) | 347.6 (3) | 0.99 |
| new | `bmm_mm_f1d4bb634587` window attn bwd, 4 stages | mm -> bmm -> sum -> bmm | 836.1 | 825.1 (3) | 1199.9 (6) | 0.99 |
| new | `addmm_2bce69bbfd51` proj -> LN(192) -> fc1 (N=768) | addmm -> var_mean -> addmm | 74.9 | 72.1 (3) | 76.8 (4) | 0.96 |
| new | `addmm_8e39c766aedd` LN(192) -> qkv (N=576) | var_mean -> addmm | 49.6 (K-tiled 47.3) | 37.1 (2) | 32.6 (2) | **0.75** |
| new | `bmm_c044adeba521` softmax bwd -> dq, dk (16 slices) | sum -> bmm -> bmm | 19.0 | 12.8 (3) | 15.8 (5) | **0.68** |
| old | `mm_d5c06549bc59` linear K=384 -> softmax(9) | (new spec: R = 384) | 15.9 | 18.1 (2) | 20.2 (2) | **1.14** |
| old | `bmm_2179fd3bf388` q k^T K=256 -> softmax | (new spec: R = 256) | 9.2 | 9.9 (2) | 16.1 (2) | 1.07 |
| old | `mm_4e6754a86aad` mm K=768 -> column sum | (new spec: R = 768) | 8.1 | 7.7 (2) | 8.6 (2) | 0.94 |
| old | `bmm_mm_9519247d9dd3` q-proj K=768 -> scores | (new spec: R = 768) | 545.8 | 509.3 (2) | 717.5 (4) | 0.93 |
| old | `addmm_bmm_e2c829b4a8ca` q AND k proj -> scores | (new spec: F = 32768) | 1027.3 | 842.4 (3) | 1026.2 (6) | **0.82** |

`saved ratio` below = bytes a one-kernel version avoids moving (each
stage-to-stage tensor written and re-read, or only re-read if it is also an
output) / bytes of the region's external inputs and outputs.

| repro | saved ratio | independent slices | fused/unfused |
| --- | --- | --- | --- |
| `bmm_mm_3759ef405a77` | 0.015 | 16 x 256/BD | 1.05 |
| old `bmm_mm_9519247d9dd3` | 0.048 | many | 0.93 |
| `bmm_1d7e3367c27c` | 0.055 | 589824 | 1.09 |
| `mm_83df85014e51` | 0.062 | 131072 | 0.99 |
| `bmm_c044adeba521` | 0.16 | **16** | 0.68 |
| `addmm_bmm_86f50de0a77b` | 0.17 | many | 1.23 |
| `bmm_mm_52f22734ce29` | 0.20 | 32768 | 1.18 |
| `bmm_e90e4276642b` | 0.23 | 4096 groups | 1.38 |
| `bmm_a7f1e10d4357` | 0.38 | 192 groups | 1.19 |
| `addmm_bmm_478c63b3f825` | 0.65 | 32768 | 1.11 |
| `addmm_2bce69bbfd51` | 0.71 | 25216 | 0.96 |
| `addmm_8e39c766aedd` | 0.80 | 25216 | 0.75 |
| `bmm_mm_f1d4bb634587` | 0.87 | 32768 x 49 | 0.99 |
| `bmm_ac33030a7533` | 1.90 | 65536 | 1.22 |
| `bmm_0f0f8a25f92c` | 3.60 | 65536 | 1.48 |
| `addmm_bmm_cd12c211d331` | 9.49 | 131072 | 2.03 |

## Findings

1. **Feasibility.** Every new-corpus vertical repro that was written became
   exactly one kernel without recomputation; none failed to compile. The
   extractor's slice/F analysis is a sound feasibility test. In the old
   corpus, `e2c8` (both bmm operands are projections inside the region) only
   fuses by holding a whole head's 512 x 64 keys per program: 0.82x, and
   several autotune configs exceed shared memory. That family is not fusable
   in practice; the old partitioner captures it, the new F check rejects it.
2. **R of the first MM does not limit fusion.** K of an MM over region
   inputs is streamed from memory, so the old-corpus regions with K = 256 to
   768 all fused (`d5c0` 1.14x, `2179` 1.07x). The two that did not win
   (`9519`, `4e67`) lost for reasons 3a/3c below, not because of K.
3. **When one kernel is slower** (all other cases won, up to 2.03x):
   - a. **Parallelism collapse.** dq reads rows and dk reads columns of the
     softmax-backward `ds`, so the slice is a whole head: 16 slices on
     132 SMs gave 0.68x; the same pattern with 192 / 512 / 4096 slices gave
     1.19x / 1.21x / 1.38x. Tiny regions (`1798`, 16 slices) still win
     because their unfused kernels are just as small.
   - b. **Serial N after a reduction on the MM's K axis.** When a row
     reduction (LN statistics, softmax) feeds an MM, avoiding recomputation
     means one program computes all N columns of its rows in a serial loop.
     Attention's P @ V (N = head_dim 128-256) wins; LN(192) -> N = 576
     loses (0.75x, a K-tiled variant without power-of-two padding 0.78x, so
     padding is not the cause) and N = 768 is neutral (0.96x). N = 512 FFNs
     won only with large savings (`52f2`, `478c`).
   - c. **Little to save.** With saved ratio < ~0.1 fusion is 0.93-1.05x
     (`9519`, `83df`, `3759`), unless the region is tiny (launch-bound).
   - d. **Large per-program working set.** `f1d4` (4 stages on 49 x 49
     tiles padded to 64, most intermediates also outputs) saves 87% of the
     external traffic but runs 0.99x.

## Pattern extractor vs the previous partitioner (vertical)

| | new (R, F <= 192, slices) | old (first MM any K, chained K <= 128, reductions <= 128) |
| --- | --- | --- |
| captures infeasible regions | no | yes: both bmm operands internal (`e2c8`, the `horizontal_then_mm` family) |
| captures slow regions | yes: few slices (`c044`), reduction -> large-N MM (33 repros incl. LN(192) -> 576/768 and LN -> vocab N = 30528) | yes: feasible but unprofitable (`9519`, `4e67`) |
| misses fusable, faster regions | yes: first MM with K > 192 (`d5c0`, `2179` and the projection -> small-reduction families) | LN/softmax over 129-192 (measured unprofitable anyway) |

## Suggested heuristic changes

1. Keep the slice/F analysis as the feasibility test (count F at the
   power-of-two size Triton pads to).
2. Drop R for reductions whose reduced operand streams from region inputs
   (the first MM's K, norms over inputs); R matters only through F.
3. Parallelism: require at least as many independent slices as SMs (132 on
   H100), unless the region is launch-bound (external bytes below ~1 MB).
   On the new corpus this drops `c044` only (the other 12 regions with
   fewer than 132 slices are tiny).
4. Serial N: if an MM's K axis is fed by an internal reduction, the product
   of its output axes outside the slice should be <= 256; 512-768 only with
   a high saved ratio; never for vocabulary-sized N. 33 new repros (116
   occurrences) have N > 256 there.
5. Profitability: require saved ratio >= ~0.1 for non-tiny regions.

Caveats: quick autotuning only, one shape point per repro, hand-written
kernels (fused and unfused alike), unfused = Helion per-stage kernels
(cuBLAS-backed Inductor is listed separately), H100 only.

## Applied

The simplest change was taken: the vertical limits were lowered from 192 to
128 (`MatmulLimits`), and the corpus keeps one repro per family (structure +
input/output dtypes). The previous partitioner and its corpus were removed;
their five kernels stay in `benchmarks/vertical_helion/old/`. Of the measured
repros, `addmm_8e39c766aedd`, `addmm_2bce69bbfd51` and `mm_83df85014e51` are
no longer extracted (R = 144-192), and `addmm_41b6b1a7d236` lost its C = 184
shape point (re-measured on C = 64: 1.33x,
`helion_vertical_result_c64.json`). `bmm_c044adeba521` (16 slices) is still
extracted; the parallelism rule (3 above) would drop it.
