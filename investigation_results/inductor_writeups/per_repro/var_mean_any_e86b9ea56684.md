# var_mean_any_e86b9ea56684


## Measured Timings
- Oracle: 8.29 us
- Compile (CDT): 9.25 us
- Ratio: 1.12x

- **Gap**: 1.07-1.14x (originally reported 2.02x -- mostly FAKE gap from dispatch overhead)
- **Classification**: RECLASSIFIED (borderline)
- **Root cause**: Without CUDAGraph, dynamo dispatch overhead dominated. With CUDAGraph the gap is only 1.07-1.14x depending on the run, which is borderline/noise. A split-reduction-for-fusion pass was implemented in the scheduler (splits flat `any` into per-row any + final) which is architecturally useful for other cases.
- **Fix approach**: Split-reduction-for-fusion pass already implemented. This specific repro does not benefit meaningfully.
- **Status**: RECLASSIFIED
- **Affected repros**: 1 (borderline; pass may help other repros)
