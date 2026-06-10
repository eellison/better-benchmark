# sum_sum_63e248035ceb (GhostNet BN-backward, [512,160,7,7] channels-last)

## Status: CLOSED — AT_FLOOR (2026-06-10)

## Measurements (B200, GPU 0)
| | oracle_us | compile_us | ratio |
|---|---|---|---|
| Before (pr-184905 baseline) | 21.1-21.3 | 35.6-35.7 | 1.68x |
| After fix (bdc289b3644) | 23.4 | 24.1 | **1.03x AT_FLOOR** |

## Reduction shape
- Two sibling sum([0,2,3]) over sliced [512,80,7,7] view: xnumel=80, rnumel=512*7*7=25088
- Plus full [512,80,7,7] channels-last epilogue output + [80] vector
- Graph has a channels-last copy producer (new_empty_strided + copy) feeding the slice

## Diverging decision vs at-floor sibling sum_sum_e9338369070e
- Sibling: xnumel=40, rnumel=100352 -> aggressive split path fires (split=30, 1200 CTAs)
- THIS repro: xnumel=80, rnumel=25088 -> reduction_split_factor returned 1
- File:line: `/tmp/pytorch-work/torch/_inductor/choices.py:570-576` (pre-fix numbering):
  the `split_reductions_for_undersaturated_gpu` no_split_threshold lowering was gated on
  `numel_hint < num_sm // 2` = 74 on B200. xnumel=80 >= 74, so no_split_threshold stayed
  at the B200 value 524288; rnumel 25088 <= 524288 -> return 1 (no split).
- Consequence: single fused reduction kernel (80 CTAs on 148 SMs, 54% block occupancy)
  with the FULL [512,80,7,7] epilogue store fused as a second r-loop in the same kernel.
  Sibling instead gets partial-reduce + finalize + fully-parallel pointwise epilogue.

## Classification (design TODO #1)
**(a) split not firing.** The split machinery (standard heuristic split=4) closes the gap
once the gate admits xhint=80. Note the secondary effect: enabling the split un-fuses the
full-size epilogue into its own saturated pointwise kernel — here that fusion break is the
WIN (the fused epilogue was serialized over 80 CTAs), i.e. the opposite of the cancel-split
case (b). Reinforces TODO #1: the split decision and the epilogue-fusion decision are one
joint structure decision.

## Exploration (in-process monkeypatch, pre-fix)
| Mode | compile_us |
|------|-----------|
| baseline (split=1) | 35.58 |
| force_cooperative_reductions | 37.82 |
| split=2 | 28.45 |
| split=4 (standard heuristic) | **26.27** |
| split=6 | 30.34 |
| split=8 | 28.19 |
| split=15 (aggressive path value) | ~29.8 |
| split=30 | 29.09 |

Standard split factor (4) beats the aggressive 8-wave value -> only the threshold gate was
widened; the aggressive path stays at `< num_sm // 2`.

## Fix (commit bdc289b3644 on pr-184905)
- `torch/_inductor/config.py`: new flag `split_reductions_for_partially_saturated_gpu = True`
- `torch/_inductor/choices.py` reduction_split_factor: extend the no_split_threshold
  lowering to `num_sm // 2 <= numel_hint < num_sm`, with an extra work gate
  `reduction_numel_hint * numel_hint <= min_elements_per_device` (9.7M on B200).
  Without the work gate, densenet sum_sum_57e5518c4d1d (xhint=96, rnumel=200704,
  work=19.3M) regressed 85.7us -> 91.7us; with it, that repro stays unsplit at 1.01x.

## Post-fix family / guard sweep (B200, fresh cache each)
| repro | before | after | note |
|---|---|---|---|
| sum_sum_63e248035ceb | 1.68x | **1.03x AT_FLOOR** | target |
| sum_sum_57e5518c4d1d | 1.05x | 1.01x | guard; work gate keeps unsplit |
| sum_sum_e9338369070e | 0.68x | 0.70x | sibling, unchanged path |
| var_mean_598830735cf6 | 1.40x | 1.42x | guard; xhint=64 < num_sm//2, unchanged path |
| sum_sum_b16afac198fb | 0.97x | 0.97x | guard |
| sum_sum_sum_f0377fc40fe2 | 1.39x | 1.39x | xhint=192 > num_sm, outside window |
| sum_sum_sum_3ec568a7ba04 | 1.21x | 1.23x | xhint=1536, outside window |
| sum_sum_sum_335abef71e93 | BENCH_FAILED | BENCH_FAILED | pre-existing harness arity bug (oracle_full() takes 7 args, given 11) |

## Remaining gap members (separate investigations)
f0377fc40fe2 (1.39x) and 3ec568a7ba04 (1.23x) have numel_hint above num_sm — their gaps are
NOT this gate; different diverging decision needed.
