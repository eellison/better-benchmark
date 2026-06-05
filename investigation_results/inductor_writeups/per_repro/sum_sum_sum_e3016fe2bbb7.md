# sum_sum_sum_e3016fe2bbb7

## Classification: MULTI_OUTPUT_SHARED_REDUCTION

## Oracle: oracle_multi_output_reduction.py

## Measurements

- Compiled (default + combo + cd): 7.9 us
- Compiled (multi_kernel=1): 56.7 us
- Compiled (multi_kernel=2): 55.4 us
- Compiled (multi_kernel=3): 57.1 us
- Oracle: 7.0 us
- Ratio: 1.133x
- Oracle correctness: PASS

## Diagnosis

The oracle computes a multi-output reduction for shapes [2, 128] and [2], performing all reductions in a single kernel pass. Inductor with combo_kernels + coordinate_descent_tuning gets close (7.9 us vs 7.0 us oracle = 1.133x gap), but multi_kernel settings are dramatically worse (56-57 us), suggesting they disable the combo_kernels optimization that is critical here.

The gap is small (0.9 us absolute) and within the noise floor for such small kernels. The combo_kernels fusion is already handling most of the fusion opportunity.

## Config exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (combo + cd) | 7.9 | 1.133x |
| multi_kernel=1 | 56.7 | 8.1x |
| multi_kernel=2 | 55.4 | 7.9x |
| multi_kernel=3 | 57.1 | 8.2x |

Best config is default combo + cd. Multi_kernel disables combo_kernels fusion and is catastrophic.

## Root cause

For very small reduction shapes ([2, 128] -> [2]), the overhead is mostly kernel launch latency. The oracle eliminates one kernel launch by fusing both outputs into a single program. Inductor with combo_kernels gets close but still has a marginal scheduling overhead.

## Fix path

- Minor: combo_kernels could fuse the final [2] reduction with the [2, 128] reduction output
- Low priority given the 0.9 us absolute gap

## Status: at_floor (1.13x gap, 0.9us absolute, combo+cd is best)
