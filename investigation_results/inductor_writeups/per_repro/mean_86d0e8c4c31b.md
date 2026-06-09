# mean_86d0e8c4c31b


## Measured Timings
- Oracle: 339.90 us
- Compile (CDT): 344.06 us
- Ratio: 1.01x

## Classification: AT_FLOOR

## Pattern

RMSNorm forward (bf16[1152000, 512]): Inductor matches oracle within 2%.

## Measurements

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Oracle | 339.90 | 1.000 |
| torch.compile | 345.86 | 1.018 |

Correctness: PASS (shape=[1152000, 512] bf16, max_diff=3.12e-02)

## Diagnosis

Ratio 1.018x is at noise floor. Inductor's persistent reduction kernel for
standalone RMSNorm (without downstream global sum) is already optimal.

## Status: AT_FLOOR - no action needed
