# sum_sum_sum_e4f9a61a4b53

## Classification: MULTI_OUTPUT_SHARED_REDUCTION

## Oracle: oracle_multi_output_reduction.py

## Measurements

- Compiled (default + combo + cd): 16.6 us
- Compiled (multi_kernel=1): 58.0 us
- Compiled (multi_kernel=2): 60.2 us
- Compiled (multi_kernel=3): 60.3 us
- Oracle: 15.4 us
- Ratio: 1.083x
- Oracle correctness: PASS

## Diagnosis

The oracle computes a multi-output reduction for shapes [128, 4096] and [128], performing all reductions in a single kernel pass. Inductor with combo_kernels + coordinate_descent_tuning achieves 16.6 us (1.08x gap), while multi_kernel settings are dramatically worse (~60 us), indicating they break combo_kernels fusion.

The gap is small (1.2 us absolute) and approaching the noise floor. Combo_kernels is already handling the key fusion.

## Config exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (combo + cd) | 16.6 | 1.083x |
| multi_kernel=1 | 58.0 | 3.8x |
| multi_kernel=2 | 60.2 | 3.9x |
| multi_kernel=3 | 60.3 | 3.9x |

Best config is default combo + cd. Multi_kernel disrupts combo_kernels and is much worse.

## Root cause

The oracle fuses the [128, 4096] -> [128] column reduction with a dependent [128] -> scalar (or similar) operation in one kernel. Inductor with combo_kernels gets close but has marginal overhead from scheduling multiple fused subkernels vs. the oracle's monolithic approach.

## Fix path

- Minor optimization in combo_kernel scheduling overhead
- Low priority given the 1.2 us absolute gap and < 1.1x ratio

## Status: at_floor (1.08x gap, 1.2us absolute, combo+cd is best)
