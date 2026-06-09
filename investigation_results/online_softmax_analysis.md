# Online Softmax Large Row Optimization Analysis

## Target Repros

| Repro | Pattern | Shape | total_bytes | n_kernels |
|-------|---------|-------|-------------|-----------|
| `amax_sum_sum_dc96a87ba8db` | softmax_bwd (amax+exp+sum+div+bf16_cast+sum) | bf16[8192, 262144] -> scalar | 4.0 GB | 2 |
| `amax_sum_3ed297ef02cd` | softmax_fwd (amax+sub+exp+sum+div+bf16_cast) | bf16[8192, 262144] -> bf16[8192, 262144] | 8.0 GB | 1 |
| `amax_sum_f0661488d68c` | cross_entropy_fwd (log_softmax+gather) | bf16[8192, 262144] + i64[8192] | 4.0 GB | 1 |

All three share: numel=8192, rnumel=262144, bf16 dtype, reduction over dim=-1.

## Current Timing (3-config interleaved sweep, baseline branch)

| Repro | default (us) | combo_persistent (us) | combo_looped (us) | memcopy_SOL (us) |
|-------|-------------|----------------------|-------------------|------------------|
| dc96a87ba8db | 3541 | 3191 | 3192 | 624 |
| 3ed297ef02cd | 3730 | 3469 | 3537 | 1244 |
| f0661488d68c | 2261 | 2225 | 2224 | 628 |

Gap to SOL: 5.7x, 3.0x, 3.6x respectively (best config).

## With Stash Optimizations (sweep_all_fixes)

| Repro | stash default (us) | stash coord_descent (us) | speedup vs baseline best |
|-------|-------------------|-------------------------|--------------------------|
| dc96a87ba8db | (not measured in sweep_all_fixes) | -- | -- |
| 3ed297ef02cd | 2733 | 2680 | 1.29x vs 3469 |
| f0661488d68c | 1450 | 1290 | 1.72x vs 2224 |

Additionally, `scalar_acc_targeted` sweep (coord_descent only, no online softmax combo):
- 3ed297ef02cd: 2811 us (1.24x vs 3469 baseline best)
- f0661488d68c: 1241 us (1.79x vs 2224 baseline best)

## Oracle Floor

- dc96a87ba8db: Oracle scaffold exists (`oracle_softmax_sum.py`), measures a hand-tuned Triton online softmax with 2-pass (online row sums + scalar final reduction). Not yet measured on hardware.
- Heuristic benchmark (isolated online softmax, same shape): 2205 us for numel=8192, rnumel=262144, bf16.

## What the Stash Does for Online Softmax

The stash in `/tmp/pytorch-work` (stash@{0}) contains these relevant changes:

### 1. Scalar Reduction Accumulators (`config.triton.scalar_reduction_accumulators`)
**Files:** `torch/_inductor/codegen/triton.py`, `torch/_inductor/config.py`

- Adds a new config flag `scalar_reduction_accumulators` (default ON).
- For simple associative reductions (sum, max, min, prod, xor_sum, any), instead of keeping a full `[XBLOCK, R0_BLOCK]` tile accumulator across loop iterations, reduces each loaded tile to a scalar per x-element within each iteration.
- Result: accumulator is `[XBLOCK]` instead of `[XBLOCK, R0_BLOCK]`. Dramatically reduces register pressure.
- Enables much larger R0_BLOCK (2048/4096/8192/16384) without register spilling.
- For `online_softmax_reduce`, a specialized scalar variant also accumulates max and sum as scalars per iteration (block-level max + correction + block-level sum).

### 2. R0_BLOCK Expansion Configs
**File:** `torch/_inductor/runtime/triton_heuristics.py`

- For INNER reductions with rnumel >= 32768, adds configs with R0_BLOCK = 2048, 4096, 8192, 16384 (with num_warps=8).
- Also adds these larger configs in the general (non-hinted) path for rnumel >= 32768.
- Removes the `MAX_R0_BLOCK = 1024` cap for Blackwell, restoring it to 1024 (unchanged) but adding larger candidates for the autotuner.

### 3. Persistent Threshold Raise
**File:** `torch/_inductor/choices.py`

- Raises the persistent reduction threshold from 1024 to 2048 for INNER reductions.
- This means more kernels use the looped (non-persistent) path, which benefits from the scalar accumulator + large R0_BLOCK combination.

### 4. num_warps=2 for Persistent Reductions
**File:** `torch/_inductor/runtime/triton_heuristics.py`

- Changes `num_warps` from 1 to 2 when `rnumel >= 1024` in persistent reduction configs.
- More warps for larger rows improves ILP within a single block.

### 5. Online Softmax rnumel Threshold
**Files:** `torch/_inductor/config.py`, `torch/_inductor/lowering.py`, `torch/_inductor/fx_passes/post_grad.py`

- Adds `config.online_softmax_rnumel_threshold` (default None = never disable).
- When set, disables online softmax for reduction dims exceeding the threshold, falling back to 3-pass.
- BUT: this is only relevant when scalar accumulators are DISABLED. With scalar accumulators enabled, online softmax is always beneficial (heuristic data confirms: 2205 vs 2930 us for this exact shape).

### 6. `prepare_softmax_extra_check` Enhancement
**File:** `torch/_inductor/fx_passes/post_grad.py`

- Enhanced to check rnumel when scalar accumulators are disabled; skips online softmax conversion for rnumel >= 65536 in that case.
- When scalar_reduction_accumulators is ON, always allows online softmax (which is the desired behavior for our target repros).

### 7. Coordinate Descent Always for Mix-Order Reductions
**File:** `torch/_inductor/runtime/triton_heuristics.py`

- Forces coordinate descent tuning for `is_mix_order_reduction` kernels (regardless of config).

## Mechanism of Improvement

For these target repros (rnumel=262144, INNER reduction):

1. **Current baseline**: Uses online softmax with vector accumulators. R0_BLOCK capped at 1024 on Blackwell. Each iteration keeps `[XBLOCK, 1024]` max and sum accumulators alive in registers, limiting occupancy.

2. **With stash**: Scalar accumulators reduce the online softmax accumulators from `[XBLOCK, R0_BLOCK]` to `[XBLOCK]`. This enables:
   - R0_BLOCK = 4096 or 8192 without register spill
   - Fewer loop iterations (262144 / 4096 = 64 vs 262144 / 1024 = 256)
   - Better instruction-level parallelism per iteration (more independent work per block)

3. **Coordinate descent**: Further tunes XBLOCK/R0_BLOCK after the initial autotuning, catching improvements the discrete config grid misses.

## Specific Inductor Files/Functions That Need to Change

| File | Function/Section | Change |
|------|------------------|--------|
| `torch/_inductor/config.py` | Top-level + `class triton` | Add `online_softmax_rnumel_threshold` and `scalar_reduction_accumulators` configs |
| `torch/_inductor/codegen/triton.py` | `TritonKernel` reduction codegen (~L4430) | Add scalar accumulator path for simple reductions AND online_softmax_reduce |
| `torch/_inductor/runtime/triton_heuristics.py` | `_reduction_configs()` | Add R0_BLOCK=2048/4096/8192/16384 configs for rnumel >= 32768 |
| `torch/_inductor/runtime/triton_heuristics.py` | `_persistent_reduction_configs()` | num_warps=2 when rnumel >= 1024 |
| `torch/_inductor/choices.py` | `InductorChoices` persistent threshold | Raise INNER threshold from 1024 to 2048 |
| `torch/_inductor/fx_passes/post_grad.py` | `prepare_softmax_extra_check` | Guard online softmax for large rnumel when scalar acc disabled |
| `torch/_inductor/lowering.py` | `prepare_softmax_online` | Add rnumel_too_large check gating online softmax |

## Expected Impact

| Repro | Current Best (us) | Expected with Stash (us) | Expected Speedup |
|-------|-------------------|--------------------------|------------------|
| dc96a87ba8db | 3191 | ~2200-2700 (est. from heuristic data) | 1.2-1.5x |
| 3ed297ef02cd | 3469 | 2680 (measured) | 1.29x |
| f0661488d68c | 2224 | 1241-1290 (measured) | 1.72-1.79x |

The cross-entropy forward (f0661488d68c) shows the largest improvement because:
- It has a log_softmax + gather pattern that benefits most from fewer passes
- The online softmax with scalar accumulators fuses the log(sum(exp)) computation into a single pass
- Coordinate descent tuning further optimizes the tile size

## Risks and Dependencies

1. **Correctness**: Scalar online softmax accumulators change the numerical reduction order. The floating-point result will differ slightly from the vector accumulator path. Need to verify within acceptable tolerance (rtol=1e-2 for bf16).

2. **Regression risk**: Raising persistent threshold from 1024 to 2048 may regress some smaller-rnumel kernels. The stash addresses this by always trying both paths via autotuner.

3. **Register spill on other shapes**: The R0_BLOCK expansion adds configs (2048/4096/8192/16384) to the autotuner candidate list; it does NOT force them. The autotuner will only pick them if they are faster. Low risk.

4. **Interaction with combo_kernels**: The combo_persistent/combo_looped configs already improve these repros by ~10%. The scalar accumulator optimization is orthogonal and stacks on top.

## Implementation Priority

The changes should be landed in this order:
1. **Config flags** (config.py) - low risk, no behavioral change
2. **Scalar accumulator codegen** (codegen/triton.py) - core optimization, biggest impact
3. **R0_BLOCK expansion** (runtime/triton_heuristics.py) - enables the benefit of #2
4. **Persistent threshold + num_warps** (choices.py, triton_heuristics.py) - secondary tuning
5. **Online softmax rnumel guard** (post_grad.py, lowering.py) - safety net for when scalar acc is disabled
