"""
Prototype fused gather-reduce Triton kernel for sum_adeaebad93f7.

This kernel eliminates the 784MB intermediate buffer by fusing:
1. Sign flip
2. Stereo swap gather
3. Time window gather
4. Group permutation gather
5. Source summation

All into a single kernel that reads directly from the original arg0_1 input.
"""
import time
import torch
import triton
import triton.language as tl


@triton.jit
def fused_gather_reduce_kernel(
    in_ptr,       # arg0_1: [64, 5, 2, 426888] f32, strides [4268880, 853776, 426888, 1]
    perm_ptr,     # group_indices: [16, 4, 4] i16 (flattened sort indices)
    signs_ptr,    # signs: [64, 4] f32 (pre-computed: +1 or -1)
    stereo_ptr,   # stereo_indices: [64, 4] i64
    offset_ptr,   # offsets: [64, 4] i64
    out_ptr,      # output: [64, 2, 382788] f32
    T: tl.constexpr,        # 382788
    BLOCK_T: tl.constexpr,  # tile size along time
):
    # Grid: (n_tiles_T, 64*2) where each program handles one (n64, stereo) pair
    # and a block of time elements
    pid_t = tl.program_id(0)  # time block index
    pid_nc = tl.program_id(1)  # flattened (n64, stereo) index

    n64 = pid_nc // 2
    c = pid_nc % 2

    a16 = n64 // 4
    i4 = n64 % 4

    # Time indices for this block
    t_offset = pid_t * BLOCK_T
    t_idx = t_offset + tl.arange(0, BLOCK_T)
    t_mask = t_idx < T

    # Initialize accumulator
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Sum over 4 source lanes (unrolled by Triton compiler)
    for j in range(4):
        # Load permutation index: perm[a16, i4, j]
        perm_idx = tl.load(perm_ptr + a16 * 16 + i4 * 4 + j)
        perm_idx_i64 = perm_idx.to(tl.int64)

        # actual_n = a16*4 + perm[a16, i4, j]
        actual_n = a16 * 4 + perm_idx_i64

        # Load sign for (actual_n, j)
        sign_val = tl.load(signs_ptr + actual_n * 4 + j)

        # Load stereo index for (actual_n, j)
        stereo_val = tl.load(stereo_ptr + actual_n * 4 + j)

        # effective_stereo: for c=0 use stereo_val, for c=1 use 1-stereo_val
        effective_stereo = tl.where(c == 0, stereo_val, 1 - stereo_val)

        # Load time offset for (actual_n, j)
        time_off = tl.load(offset_ptr + actual_n * 4 + j)

        # Compute address into arg0_1[actual_n, 1+j, effective_stereo, t + time_off]
        # Strides: [5*2*426888, 2*426888, 426888, 1] = [4268880, 853776, 426888, 1]
        base_addr = actual_n * 4268880 + (1 + j) * 853776 + effective_stereo * 426888 + time_off

        # Load values for all time points in this block
        addrs = base_addr + t_idx
        vals = tl.load(in_ptr + addrs, mask=t_mask, other=0.0)

        # Accumulate with sign
        acc += vals * sign_val

    # Store result
    out_addr = n64 * 2 * T + c * T + t_idx
    tl.store(out_ptr + out_addr, acc, mask=t_mask)


@triton.jit
def fused_gather_reduce_kernel_v2(
    in_ptr,       # arg0_1: [64, 5, 2, 426888] f32
    perm_ptr,     # group_indices: [16, 4, 4] i16
    signs_ptr,    # signs: [64, 4] f32
    stereo_ptr,   # stereo_indices: [64, 4] i64
    offset_ptr,   # offsets: [64, 4] i64
    out_ptr,      # output: [64, 2, 382788] f32
    T: tl.constexpr,        # 382788
    BLOCK_T: tl.constexpr,  # tile size along time
):
    """V2: Process both stereo channels together to halve grid dimension."""
    pid_t = tl.program_id(0)  # time block index
    pid_n = tl.program_id(1)  # n64 index (0..63)

    n64 = pid_n
    a16 = n64 // 4
    i4 = n64 % 4

    # Time indices for this block
    t_offset = pid_t * BLOCK_T
    t_idx = t_offset + tl.arange(0, BLOCK_T)
    t_mask = t_idx < T

    # Initialize accumulators for both stereo channels
    acc_c0 = tl.zeros([BLOCK_T], dtype=tl.float32)
    acc_c1 = tl.zeros([BLOCK_T], dtype=tl.float32)

    for j in range(4):
        perm_idx = tl.load(perm_ptr + a16 * 16 + i4 * 4 + j)
        perm_idx_i64 = perm_idx.to(tl.int64)
        actual_n = a16 * 4 + perm_idx_i64

        sign_val = tl.load(signs_ptr + actual_n * 4 + j)
        stereo_val = tl.load(stereo_ptr + actual_n * 4 + j)
        time_off = tl.load(offset_ptr + actual_n * 4 + j)

        # For c=0: effective_stereo = stereo_val
        # For c=1: effective_stereo = 1 - stereo_val
        # Base address without stereo component:
        base_no_stereo = actual_n * 4268880 + (1 + j) * 853776 + time_off

        # Load channel for c=0 (stereo_val)
        addr_c0 = base_no_stereo + stereo_val * 426888 + t_idx
        vals_c0 = tl.load(in_ptr + addr_c0, mask=t_mask, other=0.0)
        acc_c0 += vals_c0 * sign_val

        # Load channel for c=1 (1 - stereo_val)
        addr_c1 = base_no_stereo + (1 - stereo_val) * 426888 + t_idx
        vals_c1 = tl.load(in_ptr + addr_c1, mask=t_mask, other=0.0)
        acc_c1 += vals_c1 * sign_val

    # Store both channels
    out_base = n64 * 2 * T
    tl.store(out_ptr + out_base + t_idx, acc_c0, mask=t_mask)
    tl.store(out_ptr + out_base + T + t_idx, acc_c1, mask=t_mask)


def run_fused_kernel(arg0_1, group_indices_i16, signs, stereo_indices, offsets, version=1):
    """Run the fused gather-reduce kernel."""
    T = 382788
    output = torch.empty(64, 2, T, device=arg0_1.device, dtype=torch.float32)

    BLOCK_T = 1024
    n_tiles_t = (T + BLOCK_T - 1) // BLOCK_T

    if version == 1:
        grid = (n_tiles_t, 64 * 2)
        fused_gather_reduce_kernel[grid](
            arg0_1, group_indices_i16, signs, stereo_indices, offsets, output,
            T, BLOCK_T,
        )
    else:
        grid = (n_tiles_t, 64)
        fused_gather_reduce_kernel_v2[grid](
            arg0_1, group_indices_i16, signs, stereo_indices, offsets, output,
            T, BLOCK_T,
        )
    return output


def reference_pattern(arg0_1, signs_4d, stereo_4d, offsets_4d, group_indices):
    """Original eager computation (no RNG)."""
    device = arg0_1.device
    sliced = arg0_1[:, 1:, :, :]
    signed = sliced * signs_4d
    idx_expanded = stereo_4d.expand(64, 4, 1, 426888)
    g1 = torch.gather(signed, 2, idx_expanded)
    g2 = torch.gather(signed, 2, 1 - idx_expanded)
    catted = torch.cat([g1, g2], dim=2)
    iota = torch.arange(382788, device=device, dtype=torch.int64).view(1, 1, 1, 382788)
    time_idx = iota + offsets_4d.expand(64, 4, 2, 1)
    windowed = torch.gather(catted, 3, time_idx)
    viewed = windowed.view(16, 4, 4, 2, 382788)
    group_idx_expanded = group_indices.expand(16, 4, 4, 2, 382788)
    gathered = torch.gather(viewed, 1, group_idx_expanded)
    result = gathered.view(64, 4, 2, 382788).sum(dim=1)
    return result


def main():
    torch.manual_seed(42)
    device = torch.device('cuda')
    arg0_1 = torch.randn(64, 5, 2, 426888, device=device, dtype=torch.float32)

    # Pre-compute random values
    signs = (torch.randint(0, 2, (64, 4), device=device).float() * 2 - 1).contiguous()
    stereo_indices = torch.randint(0, 2, (64, 4), device=device, dtype=torch.int64).contiguous()
    offsets = torch.randint(0, 44100, (64, 4), device=device, dtype=torch.int64).contiguous()
    group_random = torch.rand(16, 4, 4, 1, 1, device=device)
    group_sorted, group_indices = torch.sort(group_random, dim=1)
    group_indices_flat = group_indices[:, :, :, 0, 0].to(torch.int16).contiguous()

    # Reference format for correctness check
    signs_4d = signs.view(64, 4, 1, 1)
    stereo_4d = stereo_indices.view(64, 4, 1, 1)
    offsets_4d = offsets.view(64, 4, 1, 1)

    with torch.no_grad():
        # Reference
        ref = reference_pattern(arg0_1, signs_4d, stereo_4d, offsets_4d, group_indices)
        torch.cuda.synchronize()

        # Fused v1
        out_v1 = run_fused_kernel(arg0_1, group_indices_flat, signs, stereo_indices, offsets, version=1)
        torch.cuda.synchronize()
        diff_v1 = (ref - out_v1).abs().max().item()
        print(f"V1 max abs diff: {diff_v1:.6g} {'PASS' if diff_v1 < 1e-4 else 'FAIL'}")

        # Fused v2
        out_v2 = run_fused_kernel(arg0_1, group_indices_flat, signs, stereo_indices, offsets, version=2)
        torch.cuda.synchronize()
        diff_v2 = (ref - out_v2).abs().max().item()
        print(f"V2 max abs diff: {diff_v2:.6g} {'PASS' if diff_v2 < 1e-4 else 'FAIL'}")

        # Benchmark
        n_warmup = 20
        n_iters = 100

        # V1
        for _ in range(n_warmup):
            run_fused_kernel(arg0_1, group_indices_flat, signs, stereo_indices, offsets, version=1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            run_fused_kernel(arg0_1, group_indices_flat, signs, stereo_indices, offsets, version=1)
        torch.cuda.synchronize()
        v1_us = (time.perf_counter() - start) / n_iters * 1e6
        print(f"Fused V1 (separate stereo): {v1_us:.1f} us")

        # V2
        for _ in range(n_warmup):
            run_fused_kernel(arg0_1, group_indices_flat, signs, stereo_indices, offsets, version=2)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            run_fused_kernel(arg0_1, group_indices_flat, signs, stereo_indices, offsets, version=2)
        torch.cuda.synchronize()
        v2_us = (time.perf_counter() - start) / n_iters * 1e6
        print(f"Fused V2 (both stereo):     {v2_us:.1f} us")

        # Reference eager
        for _ in range(n_warmup):
            reference_pattern(arg0_1, signs_4d, stereo_4d, offsets_4d, group_indices)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            reference_pattern(arg0_1, signs_4d, stereo_4d, offsets_4d, group_indices)
        torch.cuda.synchronize()
        ref_us = (time.perf_counter() - start) / n_iters * 1e6
        print(f"Reference eager:            {ref_us:.1f} us")

        print(f"\nComparison with compiled repro (2436 us):")
        best = min(v1_us, v2_us)
        print(f"  Best fused kernel:  {best:.1f} us")
        print(f"  Speedup vs compiled: {2436 / best:.2f}x")
        print(f"  Note: fused kernel excludes sort (tiny) and RNG pre-computation")

        # Memory analysis
        read_bytes = 64 * 2 * 382788 * 4 * 4  # 4 loads per output element (4 source lanes)
        write_bytes = 64 * 2 * 382788 * 4
        total_bytes = read_bytes + write_bytes
        bw_achieved = total_bytes / best  # bytes per microsecond
        print(f"\n  Read traffic:  {read_bytes / 1e6:.1f} MB")
        print(f"  Write traffic: {write_bytes / 1e6:.1f} MB")
        print(f"  Total:         {total_bytes / 1e6:.1f} MB")
        print(f"  Achieved BW:   {bw_achieved / 1e6:.1f} GB/s")
        print(f"  BW utilization: {bw_achieved / 8e6 * 100:.1f}% of B200 peak (8 TB/s)")


if __name__ == "__main__":
    main()
