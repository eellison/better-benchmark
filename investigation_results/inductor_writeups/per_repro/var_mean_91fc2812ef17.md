# var_mean_91fc2812ef17 - oracle_swin_window_reverse_layernorm


## Measured Timings
- Oracle: 99.94 us
- Compile (CDT): 123.74 us
- Ratio: 1.24x

## Classification
SCHEDULER_FUSION

## Benchmark Results
- Oracle: 100.26 us
- Inductor (cd): 122.82 us
- Ratio: 1.225x

## Config Exploration
| Config | Time (us) |
|--------|-----------|
| coordinate_descent_tuning | 122.91 |
| combo + multi_kernel=2 | 122.69 |
| combo + multi_kernel=3 | 93.54 |

**multi_kernel=3 closes the gap entirely** (93.54 us < oracle's 100.26 us). The combo kernel with multi_kernel=3 achieves 1.07x speedup over the oracle.

## Root Cause

The repro (from timm swin_base_patch4_window7_224 inference) fuses Swin window-reverse with LayerNorm:
- Input: [401408, 128] f32 (from addmm/view)
- Residual: [128, 56, 56, 128] f32 (channels-last strided, stride=(401408, 56, 1, 3136))
- Window reverse: reshape [8192, 49, 128] -> permute [batch, grid_h, grid_w, win_h, win_w, C] -> contiguous
- Residual add from channels-last tensor
- LayerNorm: hidden=128, eps=1e-5, affine
- Output: [401408, 128] f32 (contiguous flatten)

### Oracle approach:
One fused Triton row kernel that maps output rows to window-source rows via index arithmetic, loads residual from the channels-last strided tensor, computes var_mean LayerNorm with affine, and writes the contiguous output directly.

### Inductor approach (default):
Materializes the window-reverse clone and residual add as an intermediate tensor, then runs a separate var_mean LayerNorm/affine/clone kernel. With multi_kernel=3, the combo kernel fuses these stages successfully.

### Resolution:
multi_kernel=3 already solves this case (93.54 us vs oracle 100.26 us).

## Models
- timm_swin_base_patch4_window7_224_infer (2 instances)

## Fix Assessment
**SOLVED by multi_kernel=3** -- The existing combo_kernels infrastructure with multi_kernel=3 already closes this gap and beats the oracle. This pattern (window-reverse + LayerNorm fusion) should be auto-selected for Swin architectures.
