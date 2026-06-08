# var_mean_924906e66d0b - oracle_swin_window_reverse_layernorm

## Classification
NEW_PATTERN

## Benchmark Results
- Oracle: 108.32 us
- Inductor (cd): 130.72 us
- Ratio: 1.207x

## Config Exploration
| Config | Time (us) |
|--------|-----------|
| coordinate_descent_tuning | 129.98 |
| combo + multi_kernel=2 | 129.86 |
| combo + multi_kernel=3 | 95.97 |

**multi_kernel=3 closes the gap entirely** (95.97 us < oracle's 108.32 us). The combo kernel with multi_kernel=3 achieves 1.13x speedup over the oracle.

## Root Cause

The repro (from timm swin_base_patch4_window7_224 training) computes the same Swin window-reverse residual LayerNorm as var_mean_91fc2812ef17, but with an additional rsqrt / 128 side output:
- Input: [401408, 128] f32
- Residual: [128, 56, 56, 128] f32 (channels-last strided)
- Window reverse: 7x7 windows from [8192, 49, 128]
- LayerNorm: hidden=128, eps=1e-5, affine
- Output 1: [401408, 128] f32 (normalized)
- Output 2: [128, 3136, 1] f32 (rsqrt side output, used for backward)

### Oracle approach:
One Triton row kernel mapping output rows to window-source rows, computing var_mean + affine + rsqrt side output in a single pass without materializing intermediates.

### Inductor approach:
Same issue as var_mean_91fc2812ef17 -- materializes window-reverse + residual intermediate. With multi_kernel=3, this is solved.

### Resolution:
multi_kernel=3 already solves this case (95.97 us vs oracle 108.32 us).

## Models
- timm_swin_base_patch4_window7_224_train (2 instances)

## Fix Assessment
**SOLVED by multi_kernel=3** -- Same family as var_mean_91fc2812ef17. The combo_kernels with multi_kernel=3 fuses the window-reverse + LayerNorm pipeline and outperforms the hand-tuned oracle.
