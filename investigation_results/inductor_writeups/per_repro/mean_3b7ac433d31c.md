# mean_3b7ac433d31c

## Classification: AT_FLOOR

## Pattern

ReLU + mean + mask: Inductor is within 5% of oracle (0.954x, oracle slower).

## Measurements

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Oracle | 77.44 | - |
| torch.compile | 73.86 | 0.954 |

Correctness: PASS (shape=[512, 1000] f32; bool mask exact match)

## Diagnosis

Ratio < 1.0, oracle is slightly slower. Inductor is optimal.

## Status: AT_FLOOR - no action needed
