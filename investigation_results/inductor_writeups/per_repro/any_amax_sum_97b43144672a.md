# any_amax_sum_97b43144672a

## Classification: CONSTANT_FOLDING

## Current Result

- Family: `full_attention_softmax`
- Oracle path: `repros/canonical/any_amax_sum_97b43144672a/oracle_full_attention_softmax.py`
- Correctness: PASS
- Oracle: `10.02 us`
- `torch.compile coordinate_descent_tuning=True`: `11.04 us`
- Ratio: 1.102
- Status: `minor_gap`

## Diagnosis

The oracle proves the arange(512)>=0 mask is always True and the fp16 0/-inf where bias always selects zero, folding the entire mask construction/guard/expand path into a pure row-softmax kernel. Inductor lowers the decomposed view/iota/ge/expand/where/add/eq/any/amax/sub/exp/sum/div/where/expand/view graph without recognizing the data-independent mask evaluates to a constant.

The gap is 1.1x (marginal). The oracle's docstring self-classifies as BANDWIDTH_BOUND, indicating the remaining cost is dominated by the row softmax and launch floor.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 11.04 |
| combo+mk=2 | ~11 (no improvement -- CUDAGraph confounds at this timescale) |
| combo+mk=3 | ~11 |
| Oracle | 10.02 |

## Root cause

Algebraic constant folding of data-independent iota predicates. The arange >= 0 is always True, any(eq(-inf)) is always False, and the where bias is always 0. Inductor does not fold these statically.

## Kernel count
- Oracle: 1 kernel (fused row-softmax)
- Inductor: ~3 kernels (mask construction, guard reductions, softmax)

## Recommendation

Minor gap (1.1x). Could be closed by constant-folding iota predicates in `torch/_inductor/fx_passes/` but marginal benefit.
