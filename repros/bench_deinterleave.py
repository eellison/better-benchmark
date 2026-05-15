"""Compare stride-2 deinterleave vs contiguous-load + register-split."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch, time
import triton
import triton.language as tl

B, H, S, D = 4, 128, 512, 64
D_HALF = 32
N = B * H * S  # 262144


@triton.jit
def deinterleave_stride2(IN, OUT, BLOCK_Y: tl.constexpr):
    """Stride-2 loads (like what Inductor generates)."""
    D: tl.constexpr = 64
    D_HALF: tl.constexpr = 32
    N: tl.constexpr = 262144

    pid = tl.program_id(0)
    y_off = pid * BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]
    x_off = tl.arange(0, D_HALF)[None, :]

    y_mask = y_off < N

    evens = tl.load(IN + y_off * D + 2 * x_off, mask=y_mask)
    odds = tl.load(IN + y_off * D + 2 * x_off + 1, mask=y_mask)

    tl.store(OUT + y_off * D + x_off, evens, mask=y_mask)
    tl.store(OUT + y_off * D + D_HALF + x_off, odds, mask=y_mask)


@triton.jit
def deinterleave_contig(IN, OUT, BLOCK_Y: tl.constexpr):
    """Contiguous load + register-level split/reshape."""
    D: tl.constexpr = 64
    D_HALF: tl.constexpr = 32
    N: tl.constexpr = 262144

    pid = tl.program_id(0)
    y_off = pid * BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]
    d_off = tl.arange(0, D)[None, :]

    y_mask = y_off < N

    data = tl.load(IN + y_off * D + d_off, mask=y_mask)

    data_r = tl.reshape(data, [BLOCK_Y, D_HALF, 2])
    evens, odds = tl.split(data_r)

    half_off = tl.arange(0, D_HALF)[None, :]
    tl.store(OUT + y_off * D + half_off, evens, mask=y_mask)
    tl.store(OUT + y_off * D + D_HALF + half_off, odds, mask=y_mask)


if __name__ == "__main__":
    x = torch.randn(N, D, dtype=torch.bfloat16, device='cuda')
    out1 = torch.empty(N, D, dtype=torch.bfloat16, device='cuda')
    out2 = torch.empty(N, D, dtype=torch.bfloat16, device='cuda')

    BLOCK_Y = 4
    grid = ((N + BLOCK_Y - 1) // BLOCK_Y,)

    # Compile
    deinterleave_stride2[grid](x, out1, BLOCK_Y)
    deinterleave_contig[grid](x, out2, BLOCK_Y)
    torch.cuda.synchronize()

    # Verify correctness
    assert torch.allclose(out1, out2), f'Max diff: {(out1 - out2).abs().max()}'
    print('Correctness: PASS')

    # Benchmark
    def bench_kernel(launch_fn, warmup=200, iters=1000):
        for _ in range(warmup):
            launch_fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            launch_fn()
        torch.cuda.synchronize()
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6

    for BY in [4, 8, 16]:
        grid = ((N + BY - 1) // BY,)
        t1 = bench_kernel(lambda: deinterleave_stride2[grid](x, out1, BY))
        t2 = bench_kernel(lambda: deinterleave_contig[grid](x, out2, BY))
        bytes_moved = N * D * 2 * 2
        print(f'\nBLOCK_Y={BY}:')
        print(f'  Stride-2:     {t1:.1f} us ({bytes_moved/t1/1e6:.1f} GB/s)')
        print(f'  Contig+split: {t2:.1f} us ({bytes_moved/t2/1e6:.1f} GB/s)')
        print(f'  Speedup:      {t1/t2:.2f}x')

    # Memcopy baseline
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        x.clone()
    torch.cuda.synchronize()
    for _ in range(200):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(1000):
        g.replay()
    torch.cuda.synchronize()
    t_clone = (time.perf_counter() - t0) / 1000 * 1e6
    print(f'\nMemcopy baseline: {t_clone:.1f} us')
