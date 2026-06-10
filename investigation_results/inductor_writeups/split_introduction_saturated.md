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
