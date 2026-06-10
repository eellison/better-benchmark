# sum_sum_63e248035ceb (GhostNet BN-backward, [512,160,7,7] channels-last)

## Status: IN PROGRESS (2026-06-10)

## Measurements (B200, GPU 0)
- Oracle: 21.25 us
- Compile (baseline, branch pr-184905): 35.62 us -> ratio 1.676x

## Reduction shape
- Two sibling sum([0,2,3]) over sliced [512,80,7,7] view: xnumel=80, rnumel=512*7*7=25088
- Plus full [512,80,7,7] channels-last epilogue output + [80] vector

## Diverging decision vs at-floor sibling sum_sum_e9338369070e
- Sibling: xnumel=40, rnumel=100352 -> reduction_split_factor returns 30 (aggressive split path)
- PRIMARY: xnumel=80, rnumel=25088 -> reduction_split_factor returns 1
- File:line: /tmp/pytorch-work/torch/_inductor/choices.py:567-578 (`no_split_threshold` lowering
  gated on `numel_hint < num_sm // 2` = 74 on B200). xnumel=80 >= 74, so no_split_threshold stays
  at the B200 value 524288; rnumel 25088 <= 524288 -> return 1 (no split).
- Consequence: single fused reduction kernel (xnumel=80 grid -> only 80 CTAs on 148 SMs, 54% util)
  with the FULL [512,80,7,7] epilogue store fused as a second r-loop in the same 80-CTA kernel.
  Sibling instead gets partial-reduce (1200 CTAs) + finalize + separate full-parallelism pointwise
  epilogue: 4 kernels but each saturated.

## Exploration (in-process monkeypatch of reduction_split_factor for (80,25088))
| Mode | compile_us |
|------|-----------|
| baseline (split=1) | 35.58 |
| force_cooperative_reductions | 37.82 |
| coop widened to xhint<=128 | 35.58 (does not trigger: rnumel 25088 < 65536 threshold) |
| split=2 | 28.45 |
| split=4 | **26.27** |
| split=6 | 30.34 |
| split=8 | 28.19 |
| split=16 | 29.79 |
| split=30 | 29.09 |

Best so far: split=4 -> 26.3us (1.24x). Cooperative is NOT the answer (xhint=80 with small rnumel).

## Next
- Test forced split on family members f0377 (192,25216) and on guard 57e5518 (96,200704 - must not regress)
- Fix location decision: heuristic gate is in LOCKED choices.py; call-site override possible in ir.py num_splits
