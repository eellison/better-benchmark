# var_mean_8ca6d493f065 - oracle_swin_window_reverse_droppath_layernorm


## Measured Timings
- Oracle: 30.24 us
- Compile (CDT): 32.51 us
- Ratio: 1.08x

## Classification
NEW_PATTERN

## Benchmark Results
- Oracle: 30.30 us
- Inductor (cd): 33.44 us
- Ratio: 1.103x

## Config Exploration
| Config | Time (us) |
|--------|-----------|
| coordinate_descent_tuning | 32.29 |
| combo + multi_kernel=2 | 31.58 |
| combo + multi_kernel=3 | 26.14 |

**multi_kernel=3 closes the gap entirely** (26.14 us < oracle's 30.30 us). The combo kernel with multi_kernel=3 achieves 1.16x speedup over the oracle itself.

## Root Cause

The repro (from timm swin_base_patch4_window7_224 training) computes Swin window-reverse, dynamic index roll, seeded drop-path residual, and LayerNorm in one scope:
- Input: [25088, 512] f32 (from addmm)
- Residual: [128, 14, 14, 512] f32 (channels-last strided)
- Window reverse: 7x7 windows -> 14x14 spatial
- Index roll: dynamic shift via int64 index tensors
- Drop-path: seeded Inductor RNG with keep_prob=0.909
- LayerNorm: hidden=512, eps=1e-5, affine
- Outputs: [25088, 512] f32 + rsqrt/512 side output

### Oracle approach:
One fused Triton row kernel that maps output rows directly to window-source rows, threads the Inductor RNG from the seed tensor for drop-path, computes the full LayerNorm, and emits both outputs.

### Inductor approach (default):
Materializes the window-reverse clone, index roll, drop-path, and residual producers before scheduling a generic var_mean LayerNorm. With multi_kernel=3, the combo kernel infrastructure successfully fuses these stages.

### Resolution:
multi_kernel=3 already solves this case -- no Inductor scheduler change needed. The combo kernel with per-subkernel blocks achieves better performance than the hand-written oracle.

## Models
- timm_swin_base_patch4_window7_224_train (2 instances)

## Fix Assessment
**SOLVED by multi_kernel=3** -- The existing combo_kernels infrastructure with multi_kernel=3 already closes this gap. This should be enabled by default for Swin-family models or detected via cost model.
