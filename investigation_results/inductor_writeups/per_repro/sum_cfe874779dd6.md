# sum_cfe874779dd6 - oracle_gptneo_ce_backward_dual_pad

## Classification
ALGEBRAIC_ELIMINATION

## Benchmark Results
- Oracle: 573.22 us
- Inductor (cd): 927.65 us
- Ratio: 1.618x

## Config Exploration
| Config | Time (us) |
|--------|-----------|
| coordinate_descent_tuning | 927.65 |
| combo + multi_kernel=2 | 926.50 |
| combo + multi_kernel=3 | 925.50 |

multi_kernel configs have negligible effect (~0.2% improvement).

## Root Cause

The repro (from hf GPTNeoForCausalLM training) computes the complete cross-entropy backward with dual padded outputs:
- Labels: shifted from arg418[:, 1:] -> [32, 128] -> flattened [4096]
- Logits: [4096, 50257] f32
- Row shifts: f32[4096, 1] (two sets)
- Output 1: [50260, 4096] f32 (transposed, padded by 3 columns)
- Output 2: [4096, 50260] f32 (row-major, padded by 3 columns)

### Oracle approach (algebraic elimination + multi-output):
The oracle replaces the materialized one-hot row reduction with the algebraically equivalent guarded label scalar (same as sum_77824d392401). Additionally, it emits both the transposed and row-major padded outputs from a single autotuned kernel that iterates over rows and columns, writing to both output layouts simultaneously with padding zeros.

### Inductor approach:
Inductor materializes the one-hot expansion, scans each 50257-wide row for the sum, then rereads rows for the exponential epilogue. It also schedules separate pad-fill kernels and a transpose copy for the dual outputs.

### Why the oracle is 1.62x faster:
1. **Algebraic elimination**: O(1) per-row label lookup vs O(50257) one-hot scan saves ~50K ops per row * 4096 rows.
2. **Multi-output fusion**: Both padded outputs are written in one pass. Inductor needs separate kernels for the pad fills and transpose.
3. **No intermediate materialization**: The one-hot tensor and pre-pad intermediates are never allocated.

## Models
- hf_GPTNeoForCausalLM_train (2 instances)

## Fix Assessment
**FX/scheduler rewrite** -- Same algebraic elimination as sum_77824d392401 (one-hot sum -> indexed scalar). Additionally, the multi-output scheduling should recognize that transposed + row-major padded outputs of the same computation can be co-emitted from one kernel.
