# Combo Kernel PID Subblock Interference

## Question

How much does one combo-kernel PID subblock affect another PID subblock?

In other words, if a combo kernel is shaped like:

```python
pid = tl.program_id(0)
if pid < num_blocks_0:
    # sub-kernel 0
elif pid < num_blocks_1:
    # sub-kernel 1
else:
    # ...
```

does sub-kernel 1 hurt sub-kernel 0 while sub-kernel 0's PIDs are running?

## Short Answer

The other PID subblocks mostly do **not** affect the currently executing block through runtime execution. A block executes only one branch, so it does not execute the other sub-kernels' memory instructions or arithmetic.

They **do** affect each other through kernel-wide launch resources:

- **Registers per thread:** fixed for the whole compiled CUDA kernel, not per branch.
- **Dynamic shared memory:** fixed for the whole compiled CUDA kernel, not per branch.
- **Physical CUDA CTA size / `num_warps`:** fixed for the whole launch, even if Inductor/Triton gives each sub-kernel its own logical tile shape such as `XBLOCK_0/RBLOCK_0` and `XBLOCK_1/RBLOCK_1`.
- **Occupancy:** derived from those kernel-wide resources, so every PID subblock inherits the worst relevant resource limits of the combined binary.

This statement applies to both sequential and round-robin combo dispatch. Dispatch policy changes which PID range runs each sub-kernel and can change temporal/cache behavior, but it does not make registers, shared memory, or physical launch block size per-branch.

For the measured `sum_sum_sum_56ca14eaee84` case, the biggest concrete cross-subblock contamination is **not additive register pressure**. The combo does have different logical block shapes per branch, but the launch still uses one physical CTA/resource envelope. The large first combo kernel launches with `256` physical threads/block and `4096` bytes dynamic shared memory for all its PIDs, whereas one important standalone branch used `128` physical threads/block and `2048` bytes dynamic shared memory.

## Measured Case

Representative repro:

```bash
python scripts/bench_compare.py \
  repros/canonical/sum_sum_sum_56ca14eaee84/repro.py \
  --config-a baseline \
  --config-b 'combo_kernels=True,combo_kernel_per_subkernel_blocks=True' \
  --gpus 0 --max-workers 1 --n-warmup 5 --n-rep 20 --rounds 1 --no-share-cache
```

On the local B200:

| Mode | Kernels | Median-ish time |
|---|---:|---:|
| Baseline | 7 | `244.8 us` |
| Combo | 3 | `355.4 us` |

So combo reduces launch count, but is about **1.45x slower** on this machine for this repro.

## NCU Evidence

Captured with Nsight Compute 2026.1.1 on B200, profiling one replay after compile/warmup.

### Baseline Kernels

| Kernel | Grid | Block | Regs/thread | Dyn smem | Occupancy limit | Active warps | DRAM % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `triton_red_fused_add_mul_permute_sub_sum_view_0` | `3072` | `128` | `48` | `2048 B` | regs=`10`, smem=`21`, warps=`16` | `52.7%` | `28.2%` |
| `triton_red_fused_add_mul_permute_sub_sum_view_1` | `96` | `128` | `28` | `128 B` | regs=`16`, smem=`28`, warps=`16` | `6.1%` | `2.1%` |
| `triton_red_fused_add_div_mul_permute_sub_sum_view_2` | `512` | `512` | `90` | `16384 B` | regs=`1`, smem=`1`, warps=`4` | `24.8%` | `44.9%` |
| `triton_red_fused_permute_sum_view_3` | `3072` | `128` | `31` | `1024 B` | regs=`16`, smem=`32`, warps=`16` | `82.5%` | `61.3%` |
| `triton_red_fused_sum_4` | `3072` | `256` | `32` | `1024 B` | regs=`8`, smem=`16`, warps=`8` | `87.2%` | `67.1%` |

### Combo Kernels

| Kernel | Grid | Block | Regs/thread | Dyn smem | Occupancy limit | Active warps | DRAM % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `triton_red_fused_0` | `19456` | `256` | `48` | `4096 B` | regs=`5`, smem=`12`, warps=`8` | `60.4%` | `23.8%` |
| `triton_red_fused_1` | `6336` | `64` | `34` | `512 B` | regs=`24`, smem=`42`, warps=`32` | `65.8%` | `60.1%` |
| `triton_red_fused_permute_sum_view_2` | `96` | `128` | `28` | `128 B` | regs=`16`, smem=`28`, warps=`16` | `6.1%` | `2.0%` |

## What This Says About Interference

### Dispatch policy matters, but for a different reason

There are three relevant dispatch shapes in the current code:

```text
SequentialDispatch:
  if pid < cumulative_blocks_0: sub-kernel 0
  elif pid < cumulative_blocks_1: sub-kernel 1

RoundRobinDispatch:
  if pid % num_kernels == 0: sub-kernel 0
  elif pid % num_kernels == 1: sub-kernel 1

SequentialFlattenGridDispatch:
  like SequentialDispatch, but flattened and with XBLOCK_0/RBLOCK_0,
  XBLOCK_1/RBLOCK_1, ... for per-subkernel blocks
```

In the local PyTorch build used for this measurement, `combo_kernel_per_subkernel_blocks=True` forces `SequentialFlattenGridDispatch`. Round-robin is only selected when `per_subkernel_blocks=False`, mixed sizes are allowed, shapes are static, and the utilization heuristic says the masked round-robin grid waste is less than about 20%.

Round-robin changes the answer in one important way: it can make sub-kernels run concurrently/interleaved across SMs instead of in large contiguous PID ranges. That can increase cache/memory-system interference if the branches touch the same tensors with incompatible access patterns. Sequential dispatch reduces that particular interleaving because all PIDs for sub-kernel 0 appear before all PIDs for sub-kernel 1 in the grid.

Round-robin does **not** change the static-resource conclusion. The whole combo kernel still has one register count, one shared-memory allocation, one physical block size, and one occupancy envelope.

### 1. Register interference exists, but was smaller than expected here

The original hypothesis was that unrelated branches might add together, e.g. `40 regs + 48 regs = 88 regs` for the combo. That is **not what happened** in the main measured combo kernel:

- Standalone first heavy reduction: `48` regs/thread.
- Combo `triton_red_fused_0`: `48` regs/thread.

So for this branch, register count did **not** inflate additively. The compiler/ptxas was able to keep the max near the heaviest branch, at least for this generated code.

But registers are still kernel-wide. If another branch had forced the compiled kernel to `64` or `90` regs/thread, every PID block in that combo kernel would inherit that larger register footprint. In this exact case, that was not the dominant mechanism for `triton_red_fused_0`.

### 2. Shared-memory and physical CTA-shape interference is very real

The standalone first heavy reduction launches as:

```text
grid=3072, block=128, regs/thread=48, dynamic_smem=2048 B
```

The combo replacement launches as:

```text
grid=19456, block=256, regs/thread=48, dynamic_smem=4096 B
```

Even with the same register count and different per-branch logical `XBLOCK/RBLOCK` values, the physical launch envelope changes. Every PID block in `triton_red_fused_0` runs under the combo kernel's physical CTA size and shared-memory allocation. That changes occupancy limits from:

```text
baseline: regs=10 blocks/SM, smem=21 blocks/SM, warps=16 blocks/SM
combo:    regs=5  blocks/SM, smem=12 blocks/SM, warps=8  blocks/SM
```

This is direct cross-subblock interference: one branch's resource/block needs constrain all branches inside the same CUDA kernel binary.

### 3. The branch ladder overhead is probably not enough by itself

Each block pays a small `pid < threshold` dispatch ladder. That cost is real, and the generated code contains it, but for these reduction kernels it is unlikely to explain a `~110 us` slowdown by itself. The larger effects are launch-resource shape and changed tiling/scheduling.

### 4. The subblocks are not executing each other's work

For a given block, inactive branches are control-flow skipped. So the mental model should not be:

```text
sub-kernel A block executes A plus dead-code work from B
```

The better model is:

```text
sub-kernel A block executes A, but inside a CUDA kernel whose fixed resources were chosen for A+B
```

That distinction matters. The interference is mostly **static launch-resource contamination**, not direct runtime memory/arithmetic contention from the other branch.

With round-robin dispatch, add one extra possible effect:

```text
sub-kernel A and B blocks are interleaved in the launch order, so they may compete
for L2/DRAM at the same time if they are resident together on the GPU.
```

That is a plausible mechanism for round-robin-specific regressions, but it is separate from the NCU result above, which was captured from `SequentialFlattenComboKernelGrid`.

## Generated-Code Details

The combo code for `sum_sum_sum_56ca14eaee84` combines reductions with different logical domains in one flattened grid:

```text
triton_red_fused_0:
  sub-kernel 0: xnumel=196608, rnumel=128, reduction_hint=OUTER
  sub-kernel 1: xnumel=32768,  rnumel=768, reduction_hint=DEFAULT
```

The generated metadata reports:

```text
grid_type = SequentialFlattenComboKernelGrid
num_kernels = 2
xnumel_0 = 196608
reduction_hint_0 = OUTER
xnumel_1 = 32768
reduction_hint_1 = DEFAULT
```

The relevant generated structure is:

```python
pid = tl.program_id(0)
x_blocks_0 = tl.cdiv(196608, XBLOCK_0)
num_blocks_0 = x_blocks_0
x_blocks_1 = tl.cdiv(32768, XBLOCK_1)
num_blocks_1 = num_blocks_0 + x_blocks_1

if pid < num_blocks_0:
    # xnumel=196608, rnumel=128
elif pid < num_blocks_1:
    # xnumel=32768, rnumel=768
```

`per_subkernel_blocks=True` gives separate logical `XBLOCK_0/R0_BLOCK_0` and `XBLOCK_1/R0_BLOCK_1`. That part is real and important. What it does **not** give is separate physical CUDA CTA size / `num_warps`, separate register count, or separate dynamic shared-memory allocation per branch.

If this same pair were emitted with `RoundRobinDispatch` instead, the source shape would look more like:

```python
if pid % 2 == 0:
    pid_offset = pid // 2
    # sub-kernel 0
elif pid % 2 == 1:
    pid_offset = pid // 2
    # sub-kernel 1
```

That would change scheduling/interleaving, but not the fact that both branches compile into one CUDA function with one physical launch-resource envelope.

## Profiling Commands

Helper script used for compile/warmup/profile separation:

```python
# /tmp/profile_combo_repro.py in this investigation
compiled = torch.compile(m, fullgraph=True, dynamic=False)
compiled(*inputs)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
compiled(*inputs)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
```

NCU command shape:

```bash
/usr/local/cuda/bin/ncu \
  --target-processes all \
  --range-filter yes:1: \
  --launch-count 20 \
  --csv --page raw \
  --metrics 'launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__block_size,launch__grid_size,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed' \
  python /tmp/profile_combo_repro.py \
    --repro repros/canonical/sum_sum_sum_56ca14eaee84/repro.py \
    --combo \
    --iters 1
```

For baseline, omit `--combo`.

## Branch-Isolated Timings

To answer which sub-kernel got slower, I cloned `triton_red_fused_0` into branch-isolated variants and timed the two branches separately.

The selected configs were:

| Work item | Logical tile | Physical `num_warps` | Regs/thread | Dyn smem |
|---|---:|---:|---:|---:|
| Baseline outer branch | `X=64, R=8` | `4` | `48` | `2048 B` |
| Baseline default branch | `X=64, R=64` | `16` | `90` | `16384 B` |
| Combo `triton_red_fused_0` | `X0=64,R0=8; X1=2,R1=1024` | `8` | `48` | `4096 B` |

Whole-kernel baseline:

| Measurement | Median time |
|---|---:|
| Baseline outer kernel | `103.488 us` |
| Baseline default kernel | `117.792 us` |
| Baseline outer + default | `221.280 us` |
| Combo full `triton_red_fused_0` | `326.656 us` |
| Combo full / baseline sum | `1.476x` |

### Branch 0: outer reduction

Branch 0 has the same logical tile as its standalone baseline, `X=64,R=8`, so this is mostly testing the cost of living inside the combo body/envelope.

| Variant | `num_warps` | Median time | Ratio vs baseline outer |
|---|---:|---:|---:|
| Baseline outer | `4` | `103.488 us` | `1.000x` |
| Branch-only clone, combo tile | `4` | `101.280 us` | `0.979x` |
| Branch-only clone, combo tile | `8` | `103.392 us` | `0.999x` |
| Branch-only clone, combo tile | `16` | `169.936 us` | `1.642x` |
| Full combo body, forced branch 0 | `4` | `148.512 us` | `1.435x` |
| Full combo body, forced branch 0 | `8` | `148.576 us` | `1.436x` |
| Full combo body, forced branch 0 | `16` | `190.528 us` | `1.841x` |
| Original combo body, prefix grid | `4` | `166.944 us` | `1.613x` |
| Original combo body, prefix grid | `8` | `146.400 us` | `1.415x` |
| Original combo body, prefix grid | `16` | `185.472 us` | `1.792x` |

Interpretation: branch 0 is not slower because its logical `X/R` tile is bad. A branch-only clone with `X=64,R=8` matches baseline. It gets slower when the other branch remains in the combo body/control-flow/resource envelope.

### Branch 1: default reduction

Branch 1 is where the combo picked a very different logical tile from standalone:

```text
standalone: X=64, R=64,  num_warps=16
combo:      X=2,  R=1024, num_warps=8
```

Measured branch-isolated times:

| Variant | Logical tile | `num_warps` | Median time | Ratio vs baseline default |
|---|---:|---:|---:|---:|
| Baseline default | `X=64,R=64` | `16` | `117.792 us` | `1.000x` |
| Combo branch 1 | `X=2,R=1024` | `4` | `207.904 us` | `1.765x` |
| Combo branch 1 | `X=2,R=1024` | `8` | `214.048 us` | `1.817x` |
| Combo branch 1 | `X=2,R=1024` | `16` | `248.896 us` | `2.113x` |
| Branch-only clone | `X=2,R=1024` | `4` | `207.904 us` | `1.765x` |
| Branch-only clone | `X=2,R=1024` | `8` | `220.224 us` | `1.869x` |
| Branch-only clone | `X=2,R=1024` | `16` | `250.880 us` | `2.130x` |
| Combo branch 1 | `X=64,R=64` | `4` | `132.128 us` | `1.122x` |
| Combo branch 1 | `X=64,R=64` | `8` | `164.992 us` | `1.401x` |
| Combo branch 1 | `X=64,R=64` | `16` | `119.712 us` | `1.016x` |
| Branch-only clone | `X=64,R=64` | `4` | `132.192 us` | `1.122x` |
| Branch-only clone | `X=64,R=64` | `8` | `130.112 us` | `1.105x` |
| Branch-only clone | `X=64,R=64` | `16` | `150.560 us` | `1.278x` |

Interpretation: branch 1 is largely slower because the combo selected a very different logical tile, `X=2,R=1024`. When branch 1 uses the standalone-preferred `X=64,R=64,num_warps=16`, the forced-branch combo-body measurement is basically baseline: `119.712 us` vs `117.792 us`.

### What the branch timings imply

For `triton_red_fused_0`, the slowdown decomposes roughly as:

| Component | Evidence |
|---|---|
| Branch 0 slowdown | `~1.4x` when measured inside full combo body, despite good `X=64,R=8` tile |
| Branch 1 slowdown | `~1.8x` with combo tile `X=2,R=1024`; mostly disappears with `X=64,R=64,num_warps=16` |
| Full combo slowdown | `326.656 us / 221.280 us = 1.476x` |

The branch 1 result is the cleanest: the combo chose a tile/warp strategy that is bad for that sub-kernel. The branch 0 result says there is also a cost from keeping branch 0 inside the full combo body/envelope, even when its logical tile matches baseline.

### Full-combo forced-config check

I also tested the stronger hypothesis: if branch 1 is bad only because it got `X=2,R=1024`, then forcing the full combo kernel to use branch 1's standalone-preferred `X=64,R=64,num_warps=16` should make the whole combo fast.

That did **not** happen in this local measurement. Full `triton_red_fused_0` timings were:

| Full combo config | Median time | Ratio vs baseline sum |
|---|---:|---:|
| Selected combo: branch 1 `X=2,R=1024,nw=8` | `324.672 us` | `1.454x` |
| Same logical tiles, `nw=4` | `349.152 us` | `1.564x` |
| Same logical tiles, `nw=16` | `417.888 us` | `1.872x` |
| Branch 1 baseline tile `X=64,R=64,nw=4` | `363.616 us` | `1.629x` |
| Branch 1 baseline tile `X=64,R=64,nw=8` | `644.160 us` | `2.885x` |
| Branch 1 baseline tile `X=64,R=64,nw=16` | `596.992 us` | `2.674x` |

So two things are simultaneously true:

- Branch-isolated branch 1 is much better with `X=64,R=64,nw=16`: `119.712 us` vs `214.048 us` with combo's selected `X=2,R=1024,nw=8`.
- Full combo is not rescued by forcing branch 1's standalone tile; in this setup it gets much slower.

This means the regression is not explained by one scalar config choice alone. The full combo body has interactions that branch-isolated timing does not capture: both branches coexist in the same compiled function, branch 0 still wants its own shape, the full flattened grid changes, and changing branch 1's tile alters the full kernel's codegen/resource behavior.

### Coordinate-descent tuning result

With `combo_kernels=True`, `combo_kernel_per_subkernel_blocks=True`, and `coordinate_descent_tuning=True`, this repro flips from a regression to a win:

```bash
python scripts/bench_compare.py \
  repros/canonical/sum_sum_sum_56ca14eaee84/repro.py \
  --config-a baseline \
  --config-b 'combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True' \
  --gpus 0 --max-workers 1 --n-warmup 10 --n-rep 100 --rounds 3 --no-share-cache
```

Result on the local B200:

| Config | Time | Relative |
|---|---:|---:|
| Baseline | `239.42 us` | `1.000x` |
| Combo + per-subkernel blocks + CD | `191.46 us` | `1.250x faster` |

CD-selected configs:

| Kernel | CD-selected config | Regs/thread | Dyn smem |
|---|---:|---:|---:|
| `triton_red_fused_0` | `X0=128,R0=8; X1=4,R1=256; num_warps=1` | `255` | `1024 B` |
| `triton_red_fused_1` | `X0=4,R0=128; X1=8,R1=64; X2=128,R2=2; X3=32,R3=4; num_warps=2` | `34` | `1024 B` |
| `triton_red_fused_permute_sum_view_2` | `X=8,R=128; num_warps=4` | `32` | `128 B` |

Per-kernel profiler breakdown for the CD combo module:

| Kernel | Avg CUDA time |
|---|---:|
| `triton_red_fused_0` | `162.466 us` |
| `triton_red_fused_1` | `20.880 us` |
| `triton_red_fused_permute_sum_view_2` | `2.528 us` |

The important change is not simply “use branch 1's standalone config.” CD finds a compromise full-combo config:

```text
non-CD branch 1 tile: X1=2, R1=1024, num_warps=8  -> ~325 us full combo kernel 0
CD branch 1 tile:     X1=4, R1=256,  num_warps=1  -> ~162 us full combo kernel 0
standalone branch 1:  X=64, R=64,    num_warps=16 -> good alone, bad when forced on full combo
```

So coordinate descent is doing the right kind of search: it is not preserving each sub-kernel's standalone optimum, but finding a joint config that works for the combined body.

## Conclusion

The different PID subblocks affect each other **substantially**, but mostly through fixed CUDA-kernel launch resources rather than by executing each other's code. If the dispatch is round-robin rather than sequential, there is an additional possible cache/DRAM interleaving effect, because different branch blocks are intentionally mixed in launch order.

For this repro, the measured cross-subblock effect is large enough to turn fewer launches into a slowdown:

```text
baseline: 7 kernels, ~245 us
combo:    3 kernels, ~355 us
```

The most important correction to the prior write-up is that **register pressure was not additive in the measured main combo kernel**. The strongest new branch-level result is that the default-reduction branch regressed mainly because combo autotuning chose `X=2,R=1024` for it instead of its standalone-preferred `X=64,R=64,num_warps=16`. The outer branch kept its preferred logical tile, but still slowed when it lived inside the full combo body/envelope.
