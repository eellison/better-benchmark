"""
Benchmark optimal Triton kernel for [B,H,S,D] -> [B,S,H,D] transpose
vs inductor-generated kernel vs eager.
"""
import torch
import triton
import triton.language as tl
import torch.utils.benchmark as benchmark


@triton.jit
def permute_bhsd_bshd_kernel(
    in_ptr, out_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Optimal kernel: each thread block handles a tile of (BLOCK_S, BLOCK_D)
    for one (B,H) pair. Reads are coalesced in the D dimension (stride-1 in input),
    and consecutive S positions give stride-D in input (good locality).
    Writes are coalesced in D dimension (stride-1 in output).
    """
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    s_offset = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
    d_offset = tl.arange(0, BLOCK_D)[None, :]

    # Input layout: [B, H, S, D] contiguous = b*H*S*D + h*S*D + s*D + d
    in_idx = b * (H*S*D) + h * (S*D) + s_offset * D + d_offset

    # Output layout: [B, S, H, D] contiguous = b*S*H*D + s*H*D + h*D + d
    out_idx = b * (S*H*D) + s_offset * (H*D) + h * D + d_offset

    mask = (s_offset < S) & (d_offset < D)
    data = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)


@triton.jit
def permute_bhsd_bshd_kernel_v2(
    in_ptr, out_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_HD: tl.constexpr
):
    """
    Alternative: tile over (S, H*D) for output coalescing.
    Each thread block handles BLOCK_S S-positions and all H*D elements.
    """
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)

    s_base = pid_s * BLOCK_S
    s_offset = s_base + tl.arange(0, BLOCK_S)[:, None]
    hd_offset = tl.arange(0, BLOCK_HD)[None, :]

    h_idx = hd_offset // D
    d_idx = hd_offset % D

    # Input: [B, H, S, D] -> b*H*S*D + h*S*D + s*D + d
    in_idx = pid_b * (H*S*D) + h_idx * (S*D) + s_offset * D + d_idx

    # Output: [B, S, H, D] -> b*S*H*D + s*H*D + h*D + d
    out_idx = pid_b * (S*H*D) + s_offset * (H*D) + hd_offset

    mask = (s_offset < S) & (hd_offset < H*D)
    data = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)


def main():
    B, H, S, D = 4, 8, 1024, 64
    inp = torch.randn([B, H, S, D], dtype=torch.float16, device='cuda')
    out = torch.empty([B, S, H, D], dtype=torch.float16, device='cuda')

    # Verify correctness of kernel v1
    BLOCK_S = 32
    BLOCK_D = 64
    grid = (S // BLOCK_S, B * H)
    permute_bhsd_bshd_kernel[grid](inp, out, B, H, S, D, BLOCK_S, BLOCK_D)
    expected = inp.permute(0, 2, 1, 3).contiguous()
    assert torch.allclose(out, expected), f"V1 mismatch! max diff: {(out - expected).abs().max()}"
    print("V1 correctness verified!")

    # Verify correctness of kernel v2
    BLOCK_HD = 512  # H*D = 8*64 = 512
    grid2 = (S // BLOCK_S, B)
    out2 = torch.empty([B, S, H, D], dtype=torch.float16, device='cuda')
    permute_bhsd_bshd_kernel_v2[grid2](inp, out2, B, H, S, D, BLOCK_S, BLOCK_HD)
    assert torch.allclose(out2, expected), f"V2 mismatch! max diff: {(out2 - expected).abs().max()}"
    print("V2 correctness verified!")

    numel = B * H * S * D
    print(f"\nData size: {numel * 2 / 1e6:.1f} MB (fp16)")
    print(f"Total transfer: {2 * numel * 2 / 1e6:.1f} MB (read + write)")
    print()

    # Benchmark V1 with different BLOCK_S
    print("=== Kernel V1 (tile over S, D for each B*H) ===")
    for bs in [4, 8, 16, 32, 64, 128]:
        if S % bs != 0:
            continue
        g = (S // bs, B * H)
        def fn(bs=bs, g=g):
            permute_bhsd_bshd_kernel[g](inp, out, B, H, S, D, bs, BLOCK_D)
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t = benchmark.Timer(stmt='fn()', globals={'fn': fn}).blocked_autorange(min_run_time=0.5)
        bw = 2 * numel * 2 / t.median / 1e9
        print(f"  BLOCK_S={bs:3d}: {t.median*1e6:.1f} us, {bw:.0f} GB/s")

    # Benchmark V2
    print("\n=== Kernel V2 (tile over S, H*D for each B) ===")
    for bs in [4, 8, 16, 32, 64]:
        if S % bs != 0:
            continue
        g2 = (S // bs, B)
        def fn2(bs=bs, g2=g2):
            permute_bhsd_bshd_kernel_v2[g2](inp, out2, B, H, S, D, bs, BLOCK_HD)
        for _ in range(10):
            fn2()
        torch.cuda.synchronize()
        t = benchmark.Timer(stmt='fn2()', globals={'fn2': fn2}).blocked_autorange(min_run_time=0.5)
        bw = 2 * numel * 2 / t.median / 1e9
        print(f"  BLOCK_S={bs:3d}: {t.median*1e6:.1f} us, {bw:.0f} GB/s")

    # Eager
    print("\n=== Eager ===")
    def eager_permute():
        return inp.permute(0, 2, 1, 3).contiguous()
    for _ in range(10):
        eager_permute()
    torch.cuda.synchronize()
    t_eager = benchmark.Timer(stmt='fn()', globals={'fn': eager_permute}).blocked_autorange(min_run_time=0.5)
    bw = 2 * numel * 2 / t_eager.median / 1e9
    print(f"  Eager: {t_eager.median*1e6:.1f} us, {bw:.0f} GB/s")

    # Inductor default
    print("\n=== Inductor (default 1D tiling) ===")
    inp_flat = inp.reshape(32, 1024, 64)

    import torch._inductor.config as inductor_config
    inductor_config.triton.prefer_nd_tiling = False

    compiled_fn = torch.compile(lambda x: x.view(4, 8, 1024, 64).permute(0, 2, 1, 3).contiguous().view(4096, 512))

    compiled_fn(inp_flat)  # warmup compile
    for _ in range(10):
        compiled_fn(inp_flat)
    torch.cuda.synchronize()
    t_ind = benchmark.Timer(stmt='fn(x)', globals={'fn': compiled_fn, 'x': inp_flat}).blocked_autorange(min_run_time=0.5)
    bw = 2 * numel * 2 / t_ind.median / 1e9
    print(f"  Inductor: {t_ind.median*1e6:.1f} us, {bw:.0f} GB/s")

    print("\n=== Summary ===")
    print(f"Best custom Triton: should be close to eager")
    print(f"Gap analysis: Inductor achieves {2*numel*2/t_ind.median/1e9:.0f} GB/s vs Eager {2*numel*2/t_eager.median/1e9:.0f} GB/s")


if __name__ == "__main__":
    main()
