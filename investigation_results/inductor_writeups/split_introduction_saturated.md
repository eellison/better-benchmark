# Split introduction for saturated-row segmented reductions (design TODO #1(b))

Regime: r-contiguous (SEGMENTED-contiguous) multi-output reductions with many
output rows (>= 2*num_sm so the existing split heuristics never fire), still
losing to a split/restructured form. Investigation started 2026-06-10.

## PRIMARY: sum_0becf9609ad7 (squeezenet avgpool-backward channel sum)

Shape: where(mask[512,1000,13,13], 0, grad[512,1000]/169).sum(dims=[0,2,3]) -> [1000]
- xnumel=1000 (>= 2*148=296 -> existing heuristic returns split=1)
- rnumel=86528 = 512*169; load idx = r0_1 + 169*x0 + 169000*r0_2
  -> SEGMENTED r-contiguity: contiguous runs of L=169 (1 byte each, bool mask),
     stride jump 169000 between batch elements.
- Fresh baseline: oracle 66.4us, compile 93.9us, 1.42x.
- CD pick: XBLOCK=2, R0_BLOCK=2048, w8, s1 (95.2us standalone).

### CD local-minimum check (task #2): NO — config space is exhausted

Standalone sweep of the EXACT generated kernel over XBLOCK{1..32} x R0{256..4096}
x w{4,8,16} x s{1,2,3}: best = 85.2us (XBLOCK=4, R0=4096, w8); CD pick 95.2us.
Only ~10us of the 29us gap is config; the rest is STRUCTURE. XBLOCK>=16
catastrophic (2.7-8ms): the [XBLOCK,R0] tile becomes a strided gather.
Root cause of slowness: with short 169-element segments, row-parallelism via
XBLOCK is unprofitable, so CTAs = 1000/XBLOCK <= 500 on 148 SMs (<= 42% thread
occupancy), latency-bound on byte loads (0.93 TB/s effective).

### Headroom A/B: force split via InductorChoices.reduction_split_factor patch

| forced split | us | note |
|---|---|---|
| 0 (=current, no split) | 95.4 | baseline |
| 2 | 83.5 | |
| 64 | 68.7 | |
| 128 | 62.4 | |
| 169 | 68.2 | chunk=512 (3.03 segments, misaligned) |
| 256 | 51.8 | |
| **512** | **46.1** | **chunk=169 = EXACT segment length** |
| 676 | 95.3 | past alignment |
| 1024 | 146.4 | |

split=512 -> phase-1 kernel xnumel=512000, r0_numel=169 PERSISTENT, load
`in_ptr0 + (r0_1 + 169*x0)` = perfectly coalesced full-stream of the mask;
phase-2 1000 rows x 512 partials. 46.1us BEATS the hand-written
cooperative-split-K oracle (66.4us). Structure win = 95.4 -> 46.1 (2.07x).

KEY RULE (confirmed on 2 repros): optimal split = rnumel / L where L = inner
contiguous segment length of the reduction loads:
- 0bec: L=169, split=86528/169=512 -> 46.1us (best of sweep)
- 21c4 (xhint=16, rnumel=1605632=128*12544, L=12544=112*112):
  current aggressive split=74 -> 57.1us; segment-aligned split=128 -> 45.9us
  (oracle 44.6us). split sweep: 32->52.3, 64->51.5, 74->57.3, 128->45.9, 256->50.2.
  Misaligned splits (74) leave div/mod in phase-1 indexing; aligned splits make
  phase-1 a contiguous streaming row reduction.

### Re-verification (2026-06-10, fresh caches, gpu-locked)

| point | writeup | re-measured |
|---|---|---|
| 0bec no-split | 95.4us | 95.3us |
| 0bec split=512 | 46.1us | 46.1us |
| 21c4 current (split=74) | 57.1us | 57.0us |
| 21c4 split=128 | 45.9us | 45.3us |

Anchors hold; proceeding to f0377/3ec568 A/B + implementation.

## Next: f0377/3ec568 structure dumps + forced-split A/B, then implement
segment-aligned split introduction in choices.py behind a new flag.

## IMPLEMENTED (2026-06-11, pytorch-work a85d79a900a)

### Where it landed

- **L-detection: `torch/_inductor/ir.py`, `Reduction.num_splits`** (the
  stride-hint loop that already classifies INNER vs OUTER, ~line 1640).
  `inner_segment_length(strides)` walks the reduction ranges innermost-out
  and accumulates L = product of the stride-1 run (stride of each next-outer
  reduction var must equal the running product, i.e. ranges [.., K, inner..]
  with inner contiguous). Computed per inner load; the hint is forwarded to
  the split heuristic only if ALL inner loads produced a segment and the
  smallest segment divides every other (shared alignment). Plumbed as a new
  optional kwarg `inner_segment_hint` on
  `InductorChoices.reduction_split_factor` (TypeError fallback for
  out-of-tree subclasses with the old signature).
- **Override: `torch/_inductor/choices.py`, `reduction_split_factor`**,
  local `segment_aligned_split(split)` wrapping all three inner-reduction
  return paths (saturated `return 1`, `no_split_threshold` `return 1`,
  standard/aggressive split returns).
- **Flag: `torch/_inductor/config.py`
  `segment_aligned_split_reductions = True`**
  (`TORCHINDUCTOR_SEGMENT_ALIGNED_SPLIT_REDUCTIONS`). No existing defaults
  changed.

### Firing condition (v1)

flag on AND inner_segment_hint = L present AND:
- 64 <= L <= 16384 (phase-1 row is a sensible persistent/short-loop block;
  anchor Ls are 169 and 12544)
- rnumel % L == 0 and K = rnumel/L > 1
- existing split is NOT already segment-aligned (kept if K % split == 0)
- rnumel > 8192 (legacy no-split threshold; two-phase not amortized below)
- xnumel * K >= 2*num_sm (phase-1 saturates)
- split = K; if K > 4096, largest divisor of K <= 4096.

### Final per-repro table (B200, fresh caches, gpu-locked)

bench.py (default / coord-desc, us) and interleaved same-process A/B of the
flag (off vs on, default configs, 3 rounds):

| repro | oracle us | bench default | bench CD | A/B off | A/B on | ratio after (best/oracle) | note |
|---|---|---|---|---|---|---|---|
| sum_0becf9609ad7 (anchor) | 66.3 | 71.8 | 41.0 | 121.7 | 74.5 | **0.62x** (was 1.42x) | split 1->512; CD XBLOCK=8 beats oracle |
| sum_21c4018e2b7b (anchor) | 44.1 | 40.7 | 39.4 | 68.7 | 43.1 | **0.89x** (was 1.21x) | split 74->128 |
| sum_sum_63e248035ceb | 21.4 | 16.4 | 16.5 | 24.4 | 23.5 | 0.77x | split 4 kept (49 % 4 != 0 but 4 aligned? no: existing split 4, K=512... L=49 < 64 -> no fire on r=25088 node; unchanged decision) |
| sum_sum_e9338369070e | 41.7 | 20.5 | 22.6 | 28.5 | 28.5 | 0.49x | L=196, existing split 512, K=512 aligned -> kept |
| sum_sum_57e5518c4d1d | 83.8 | 79.1 | 73.7 | 167.7 | 89.2 | 0.88x | split 1->64 (L=3136, K=64) |
| sum_011e69da166d | 401.4 | 441.7 | 393.4 | 438.1 | 438.2 | 0.98x | no segmented reduction; unchanged |
| var_mean_598830735cf6 | 80.8 | 84.0 | 83.9 | 119.8 | 92.0 | 1.04x (must-hold <=1.42x) | split 1->32 (L=12544, K=32) |
| sum_sum_sum_f0377fc40fe2 (untested) | 34.4 | 67.6 | 39.0 | 75.6 | 75.6 | 1.13x | OUTER reductions; heuristic untouched, no change |
| sum_sum_sum_3ec568a7ba04 (untested) | 24.5 | 24.7 | 22.9 | 32.5 | 32.5 | 0.94x | scalar path (numel_hint==1) bypasses stride scan; no hint, no change |

All MUST-HOLD sentinels pass; no gate-tightening was needed.

### Family regression sweep

24 random `[0,2,3]`-channel-sum corpus repros, interleaved off/on A/B:
geomean **1.297x faster** with the flag on, 18/24 wins, worst "regression"
0.991x (noise). Big structure wins beyond the anchors:
sum_sum_sum_1aa511fcc737 1507->377us (4.0x), sum_bd0a5d156952 383->108us
(3.6x), sum_sum_sum_fb9996974d8f 598->261us (2.3x).

Correctness: 6-case smoke (anchors, odd shapes, scalar, mean, var_mean) +
dynamic-shape recompiles all pass; test_torchinductor.py -k reduction
42 passed / 9 skipped.

Note on remaining default-config gap for 0bec: default persistent config
picks XBLOCK=4/w1 for the 512000x169 phase-1 (68us); CD finds XBLOCK=8/w1
(37us). That is a triton_heuristics persistent-config issue
(x_block = min(1024 // rnumel, 8) for rounded rnumel=256), out of scope here.
