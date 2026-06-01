"""
Benchmark: split_reduction epilogue fusion ceiling for batch norm.

Shows the speedup available if inductor could fuse the second-stage reduction
with the downstream pointwise (broadcast + normalize).

Shape: x [128, 384, 8, 8] f32, channels-last stride (24576, 1, 3072, 384)
Operation: batch norm inference-style (var_mean over [0,2,3], then normalize)

What torch.compile currently emits (3 kernels):
  K1: split reduction over spatial -> partial sums [num_splits, C]
  K2: reduce partials -> mean, var [C]
  K3: pointwise normalize [N, C, H, W]  (reads input + mean/var, writes output)

The fusion target: merge K2+K3 into one kernel, or even K1+K2+K3.
The cooperative kernel below shows the ideal: all work in one kernel launch.

Benchmark comparison:
1. torch.compile: 3 separate kernels
2. Fused cooperative Triton: 1 kernel (split reduce + barrier + normalize)
3. SOL: memcopy of same bytes
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


# --- Shape parameters ---
N, C, H, W = 128, 384, 8, 8
EPS = 1e-3
SPATIAL = N * H * W  # 8192


# --- The key kernel: cooperative split-reduce + normalize in ONE launch ---
# This mimics what inductor's split_reduction does but fuses everything.
#
# Phase 1 (split reduction): grid of blocks, each handles a [BLOCK_S, BLOCK_C]
#   tile. Computes partial sum and sum_sq, writes to workspace via atomics.
# Phase 2 (barrier + normalize): after all partials accumulated, each block
#   reads the final mean/var and normalizes its own tile in-place.
#
# Grid shape: (num_s_tiles, num_c_tiles)
# With BLOCK_S=512, BLOCK_C=64: grid = (16, 6) = 96 blocks -> good parallelism!

@triton.jit
def fused_bn_splitreduce_kernel(
    X_ptr,
    OUT_ptr,
    # Workspace buffers (pre-zeroed)
    PartialSum_ptr,   # [C] f32 - accumulates sum via atomics
    PartialSqSum_ptr, # [C] f32 - accumulates sum_sq via atomics
    Counter_ptr,      # [num_c_tiles] i32 - arrival counter per channel tile
    Done_ptr,         # [num_c_tiles] i32 - done flag per channel tile
    MeanOut_ptr,      # [C] f32 - final mean
    InvStdOut_ptr,    # [C] f32 - final inv_std
    # Constants
    C_val: tl.constexpr,
    SPATIAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NUM_S_TILES: tl.constexpr,
    EPS_val: tl.constexpr,
):
    s_tile = tl.program_id(0)
    c_tile = tl.program_id(1)

    c_start = c_tile * BLOCK_C
    s_start = s_tile * BLOCK_S

    c_offsets = c_start + tl.arange(0, BLOCK_C)
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    c_mask = c_offsets < C_val
    s_mask = s_offsets < SPATIAL

    # Load our tile [BLOCK_S, BLOCK_C] - coalesced on C (channels-last)
    offsets_2d = s_offsets[:, None] * C_val + c_offsets[None, :]
    mask_2d = s_mask[:, None] & c_mask[None, :]
    tile = tl.load(X_ptr + offsets_2d, mask=mask_2d, other=0.0)  # [BLOCK_S, BLOCK_C]

    # Phase 1: compute partial sum and sum_sq for our tile
    partial_sum = tl.sum(tile, axis=0)           # [BLOCK_C]
    partial_sq = tl.sum(tile * tile, axis=0)     # [BLOCK_C]

    # Atomically accumulate into global workspace
    tl.atomic_add(PartialSum_ptr + c_offsets, partial_sum, mask=c_mask)
    tl.atomic_add(PartialSqSum_ptr + c_offsets, partial_sq, mask=c_mask)

    # Barrier: last block to arrive computes final stats and signals
    arrived = tl.atomic_add(Counter_ptr + c_tile, 1)
    if arrived == NUM_S_TILES - 1:
        # We are the last spatial tile for this channel tile
        total_sum = tl.load(PartialSum_ptr + c_offsets, mask=c_mask, other=0.0)
        total_sq = tl.load(PartialSqSum_ptr + c_offsets, mask=c_mask, other=0.0)
        mean = total_sum / SPATIAL
        var = total_sq / SPATIAL - mean * mean
        inv_std = 1.0 / tl.sqrt(var + EPS_val)
        tl.store(MeanOut_ptr + c_offsets, mean, mask=c_mask)
        tl.store(InvStdOut_ptr + c_offsets, inv_std, mask=c_mask)
        # Signal completion
        tl.store(Done_ptr + c_tile, 1)
    else:
        # Spin until done
        while tl.load(Done_ptr + c_tile, volatile=True) == 0:
            pass

    # Phase 2: normalize our tile and write output
    mean = tl.load(MeanOut_ptr + c_offsets, mask=c_mask, other=0.0)       # [BLOCK_C]
    inv_std = tl.load(InvStdOut_ptr + c_offsets, mask=c_mask, other=0.0)  # [BLOCK_C]

    normed = (tile - mean[None, :]) * inv_std[None, :]  # [BLOCK_S, BLOCK_C]
    tl.store(OUT_ptr + offsets_2d, normed, mask=mask_2d)


def fused_bn_splitreduce(x, BLOCK_C=64, BLOCK_S=512):
    """
    Single-kernel fused batch norm using cooperative split reduction.
    This is the ideal code inductor should generate.
    """
    assert x.is_contiguous(memory_format=torch.channels_last)
    N_val, C_val, H_val, W_val = x.shape
    out = torch.empty_like(x)
    spatial = N_val * H_val * W_val

    num_s_tiles = (spatial + BLOCK_S - 1) // BLOCK_S
    num_c_tiles = (C_val + BLOCK_C - 1) // BLOCK_C

    # Pre-zeroed workspaces
    partial_sum = torch.zeros(C_val, device=x.device, dtype=torch.float32)
    partial_sq = torch.zeros(C_val, device=x.device, dtype=torch.float32)
    counter = torch.zeros(num_c_tiles, device=x.device, dtype=torch.int32)
    done = torch.zeros(num_c_tiles, device=x.device, dtype=torch.int32)
    mean_out = torch.empty(C_val, device=x.device, dtype=torch.float32)
    invstd_out = torch.empty(C_val, device=x.device, dtype=torch.float32)

    grid = (num_s_tiles, num_c_tiles)
    fused_bn_splitreduce_kernel[grid](
        x, out,
        partial_sum, partial_sq, counter, done, mean_out, invstd_out,
        C_val, spatial, BLOCK_C, BLOCK_S, num_s_tiles, EPS,
    )
    return out


# --- torch.compile baseline ---
def batchnorm_eager(x):
    """Single batch norm: var_mean + normalize."""
    var, mean = torch.var_mean(x, dim=[0, 2, 3], correction=0, keepdim=True)
    return (x - mean) / torch.sqrt(var + EPS)


batchnorm_compiled = torch.compile(batchnorm_eager)


# --- SOL: memory copy ---
def memcopy_sol(x, out):
    """Copy input to output - pure memory bandwidth bound."""
    out.copy_(x)
    return out


# --- Main benchmark ---
def main():
    torch.manual_seed(42)
    device = "cuda"

    # Create channels-last input matching the repro
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    x = x.to(memory_format=torch.channels_last)
    out_copy = torch.empty_like(x)

    print("=" * 80)
    print("Batch Norm Split-Reduction Epilogue Fusion Ceiling")
    print("=" * 80)
    print(f"Shape: [{N}, {C}, {H}, {W}] f32 channels-last")
    print(f"Strides: {x.stride()}")
    print(f"Reduction: dims=[0,2,3], {SPATIAL} elements/channel, {C} channels")
    print()

    ref = batchnorm_eager(x)

    # Test configs to find working ones
    print("--- Correctness checks ---")
    test_configs = [
        (64, 512),   # grid = (16, 6) = 96 blocks
        (64, 256),   # grid = (32, 6) = 192 blocks
        (128, 512),  # grid = (16, 3) = 48 blocks
        (128, 256),  # grid = (32, 3) = 96 blocks
        (64, 1024),  # grid = (8, 6) = 48 blocks
        (128, 1024), # grid = (8, 3) = 24 blocks
    ]
    working_configs = []
    for bc, bs in test_configs:
        num_s = (SPATIAL + bs - 1) // bs
        num_c = (C + bc - 1) // bc
        name = f"BC={bc}, BS={bs} (grid={num_s}x{num_c}={num_s*num_c})"
        try:
            out = fused_bn_splitreduce(x, bc, bs)
            diff = (ref - out).abs().max().item()
            status = "OK" if diff < 1e-3 else "FAIL"
            print(f"  {name:<45} diff={diff:.2e} [{status}]")
            if diff < 1e-3:
                working_configs.append((bc, bs, name))
        except Exception as e:
            print(f"  {name:<45} ERROR: {e}")

    if not working_configs:
        print("\nNo working fused configs! Exiting.")
        return

    # Warmup compiled version
    for _ in range(5):
        batchnorm_compiled(x)
    torch.cuda.synchronize()

    # Benchmark
    nbytes_input = x.nelement() * x.element_size()
    nbytes_output = nbytes_input
    total_bytes = nbytes_input + nbytes_output

    print(f"\nData: {nbytes_input/1e6:.1f} MB in + {nbytes_output/1e6:.1f} MB out = {total_bytes/1e6:.1f} MB")
    print()

    # SOL
    t_sol = do_bench(lambda: memcopy_sol(x, out_copy))
    # torch.compile
    t_compiled = do_bench(lambda: batchnorm_compiled(x))

    # Fused configs - with workspace allocation in the loop (realistic)
    fused_results = []
    for bc, bs, name in working_configs:
        t = do_bench(lambda bc=bc, bs=bs: fused_bn_splitreduce(x, bc, bs))
        fused_results.append((name, t, bc, bs))

    # Also benchmark the best config with pre-allocated workspace (kernel-only time)
    # to separate kernel performance from allocation overhead
    best_bc, best_bs = min(working_configs, key=lambda c: next(
        t for n, t, bc2, bs2 in fused_results if bc2 == c[0] and bs2 == c[1]))[0:2]
    # Pre-alloc workspace for the best config
    num_s_best = (SPATIAL + best_bs - 1) // best_bs
    num_c_best = (C + best_bc - 1) // best_bc
    ws_sum = torch.zeros(C, device=x.device, dtype=torch.float32)
    ws_sq = torch.zeros(C, device=x.device, dtype=torch.float32)
    ws_cnt = torch.zeros(num_c_best, device=x.device, dtype=torch.int32)
    ws_done = torch.zeros(num_c_best, device=x.device, dtype=torch.int32)
    ws_mean = torch.empty(C, device=x.device, dtype=torch.float32)
    ws_invstd = torch.empty(C, device=x.device, dtype=torch.float32)
    out_prealloc = torch.empty_like(x)

    def run_prealloc():
        # Zero the workspace (must be done each call)
        ws_sum.zero_()
        ws_sq.zero_()
        ws_cnt.zero_()
        ws_done.zero_()
        grid = (num_s_best, num_c_best)
        fused_bn_splitreduce_kernel[grid](
            x, out_prealloc,
            ws_sum, ws_sq, ws_cnt, ws_done, ws_mean, ws_invstd,
            C, SPATIAL, best_bc, best_bs, num_s_best, EPS,
        )
        return out_prealloc

    # Verify prealloc version
    out_pa = run_prealloc()
    diff_pa = (ref - out_pa).abs().max().item()
    assert diff_pa < 1e-3, f"Pre-alloc version failed: diff={diff_pa}"

    t_prealloc = do_bench(run_prealloc)

    def bw(t_ms):
        return total_bytes / (t_ms * 1e-3) / 1e9

    print(f"{'Kernel':<50} {'Time(us)':<10} {'BW(GB/s)':<10} {'vs Compile':<11} {'%SOL':<6}")
    print("-" * 87)
    print(f"  {'SOL (memcopy)':<48} {t_sol*1000:<10.1f} {bw(t_sol):<10.0f} {t_compiled/t_sol:<11.2f}x {100:<6.0f}%")
    print(f"  {'Fused (best, pre-alloc ws)':<48} {t_prealloc*1000:<10.1f} {bw(t_prealloc):<10.0f} {t_compiled/t_prealloc:<11.2f}x {t_sol/t_prealloc*100:<6.0f}%")
    for name, t, _, _ in sorted(fused_results, key=lambda x: x[1]):
        print(f"  {'Fused: ' + name:<48} {t*1000:<10.1f} {bw(t):<10.0f} {t_compiled/t:<11.2f}x {t_sol/t*100:<6.0f}%")
    print(f"  {'torch.compile (3 kernels)':<48} {t_compiled*1000:<10.1f} {bw(t_compiled):<10.0f} {'1.00x':<11} {t_sol/t_compiled*100:<6.0f}%")

    print()
    best_name, best_t_alloc = min(fused_results, key=lambda x: x[1])[:2]
    print(f"Best fused kernel: {best_name}")
    print(f"  With alloc:     {best_t_alloc*1000:.1f} us -> {t_compiled/best_t_alloc:.2f}x vs compile")
    print(f"  Pre-alloc (ws): {t_prealloc*1000:.1f} us -> {t_compiled/t_prealloc:.2f}x vs compile")
    print(f"  SOL:            {t_sol*1000:.1f} us")
    print(f"  torch.compile:  {t_compiled*1000:.1f} us")
    print()

    # The pre-alloc version is the fair comparison (inductor would statically allocate workspace)
    if t_prealloc < t_compiled:
        save_pct = (1 - t_prealloc/t_compiled) * 100
        save_us = (t_compiled - t_prealloc) * 1000
        print(f"CONCLUSION: Fused kernel is {save_pct:.0f}% faster ({t_compiled/t_prealloc:.2f}x speedup).")
        print(f"  Saves {save_us:.1f} us per batch norm by eliminating:")
        print(f"    - 2 kernel launches (~5 us each)")
        print(f"    - Intermediate buffer [C]={C} write+read (negligible bytes but launch overhead)")
        print(f"  The fused kernel achieves {t_sol/t_prealloc*100:.0f}% of memcopy SOL.")
        print()
        print(f"  For the full repro with 6 batch norms, potential savings:")
        print(f"    6 x {save_us:.1f} us = {6*save_us:.1f} us ({6*save_pct:.0f}% if all fuse similarly)")
    else:
        print(f"CONCLUSION: Fused kernel is {(t_prealloc/t_compiled-1)*100:.0f}% slower.")


if __name__ == "__main__":
    main()
