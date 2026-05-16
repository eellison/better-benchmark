"""
Optimal Triton kernel for attention head transpose: [B,H,S,D] -> [B,S,H,D]

This is the core pattern in the pointwise_4ef882a876a7 repro (1.74x inductor gap).
The repro performs 12 such transposes (from attention head outputs).

Key insight: D is the innermost (contiguous) dimension in BOTH source and dest.
- Source strides: [H*S*D, S*D, D, 1]  (B, H, S, D)
- Dest strides:   [S*H*D, H*D, D, 1]  (B, S, H, D)

This means we can tile over (S, D) with fully coalesced reads AND writes,
no shared memory transpose needed. Each program handles one (batch, head, s_block).

Design:
- 2D grid: program_id(0) = s_block, program_id(1) = batch*head
- Direct address computation (b = pid_bh // H, h = pid_bh % H)
- BLOCK_S=128 x BLOCK_D=64 tile, 8 warps
- Coalesced 128-byte reads (32 threads x 4 bytes) along D for both load/store

Also includes a fused kernel for [B,H,D,S] -> [B,S,H,D] which handles the
transposed-input case using tl.trans() to read coalesced along S and write
coalesced along D.

Results (A100, B=4, H=8, S=1024, D=64, fp16):
  Triton:   ~18 us  (493 GB/s)
  Eager:    ~16 us  (538 GB/s)
  Inductor: ~60 us  (139 GB/s)

  Triton vs Inductor: 3.4-3.5x speedup
  Triton vs Eager:    0.90x (within 10% of handwritten CUDA)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_bhsd_to_bshd_kernel(
    src_ptr, dst_ptr,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Transpose [B,H,S,D] -> [B,S,H,D] with coalesced reads and writes.

    2D grid:
      - program_id(0): s_block index (tiles along sequence dim)
      - program_id(1): batch * head (flattened outer dims)
    """
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Direct decomposition - no modular arithmetic needed beyond this
    b = pid_bh // H
    h = pid_bh % H

    # Tile offsets
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)

    # Source: [B, H, S, D] contiguous layout
    # Address: b*(H*S*D) + h*(S*D) + s*D + d
    src_base = b * (H * S * D) + h * (S * D)
    src_offsets = src_base + s_offs[:, None] * D + d_offs[None, :]

    # Dest: [B, S, H, D] contiguous layout
    # Address: b*(S*H*D) + s*(H*D) + h*D + d
    dst_base = b * (S * H * D) + h * D
    dst_offsets = dst_base + s_offs[:, None] * (H * D) + d_offs[None, :]

    # Boundary mask (D is constexpr and exactly fits BLOCK_D)
    mask = s_offs[:, None] < S

    # Load from source (coalesced: consecutive threads read consecutive d values)
    data = tl.load(src_ptr + src_offsets, mask=mask)
    # Store to dest (coalesced: consecutive threads write consecutive d values)
    tl.store(dst_ptr + dst_offsets, data, mask=mask)


@triton.jit
def _transpose_bhds_to_bshd_kernel(
    src_ptr, dst_ptr,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Transpose [B,H,D,S] -> [B,S,H,D] in one kernel (no intermediate copy).

    Source layout [B,H,D,S]: innermost dim is S (coalesced reads along S)
    Dest layout [B,S,H,D]: innermost dim is D (coalesced writes along D)

    Uses tl.trans() to handle the D<->S transposition in registers.
    """
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)

    # Source: [B, H, D, S] with strides [H*D*S, D*S, S, 1]
    # Read a [BLOCK_D, BLOCK_S] tile: src[b, h, d, s]
    src_base = b * (H * D * S) + h * (D * S)
    src_offsets = src_base + d_offs[:, None] * S + s_offs[None, :]  # [BLOCK_D, BLOCK_S]

    mask_in = s_offs[None, :] < S
    # Load [BLOCK_D, BLOCK_S] - coalesced along S (innermost in source)
    data = tl.load(src_ptr + src_offsets, mask=mask_in)

    # Transpose to [BLOCK_S, BLOCK_D] for coalesced write along D
    data_t = tl.trans(data)

    # Dest: [B, S, H, D] with strides [S*H*D, H*D, D, 1]
    dst_base = b * (S * H * D) + h * D
    dst_offsets = dst_base + s_offs[:, None] * (H * D) + d_offs[None, :]  # [BLOCK_S, BLOCK_D]

    mask_out = s_offs[:, None] < S
    tl.store(dst_ptr + dst_offsets, data_t, mask=mask_out)


def transpose_bhsd_to_bshd(src, B, H, S, D):
    """
    Transpose a [B, H, S, D] contiguous tensor to [B, S, H, D] contiguous.
    Optimal for D=64, S=1024 (attention head dimensions).
    """
    dst = torch.empty(B, S, H, D, dtype=src.dtype, device=src.device)

    BLOCK_S = 128  # Process 128 sequence positions per block
    BLOCK_D = 64   # D=64 fits exactly in one block

    grid = (triton.cdiv(S, BLOCK_S), B * H)

    _transpose_bhsd_to_bshd_kernel[grid](
        src, dst,
        H, S, D,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=2,
    )
    return dst


def transpose_bhds_to_bshd(src, B, H, D, S):
    """
    Transpose a [B, H, D, S] contiguous tensor to [B, S, H, D] contiguous.
    Fused: avoids intermediate contiguous() copy.
    """
    dst = torch.empty(B, S, H, D, dtype=src.dtype, device=src.device)

    BLOCK_S = 128
    BLOCK_D = 64

    grid = (triton.cdiv(S, BLOCK_S), B * H)

    _transpose_bhds_to_bshd_kernel[grid](
        src, dst,
        H, S, D,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=2,
    )
    return dst


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_kernel(fn, *args, warmup=25, rep=100, **kwargs):
    """CUDA event-based benchmark, returns median time in ms."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        fn(*args, **kwargs)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2]


def eager_transpose(src, B, H, S, D):
    """Eager PyTorch: permute + contiguous (uses aten::copy_)."""
    return src.view(B, H, S, D).permute(0, 2, 1, 3).contiguous()


def main():
    torch.manual_seed(42)

    # Dimensions from the repro: B=4, H=8, S=1024, D=64
    B, H, S, D = 4, 8, 1024, 64

    src = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda').contiguous()

    # Total bytes moved (read + write)
    nbytes = 2 * src.numel() * src.element_size()

    print("=" * 60)
    print(f"Attention Head Transpose: [{B},{H},{S},{D}] -> [{B},{S},{H},{D}]")
    print(f"Tensor: {src.numel() * src.element_size() / 1024**2:.2f} MB (fp16)")
    print(f"Memory traffic (r+w): {nbytes / 1024**2:.2f} MB")
    print("=" * 60)
    print()

    # --- Correctness ---
    ref = src.permute(0, 2, 1, 3).contiguous()
    out_triton = transpose_bhsd_to_bshd(src, B, H, S, D)
    assert torch.allclose(ref, out_triton), "BHSD->BSHD mismatch!"
    print("[OK] Correctness: BHSD -> BSHD")

    src_bhds = torch.randn(B, H, D, S, dtype=torch.float16, device='cuda').contiguous()
    ref_bhds = src_bhds.permute(0, 3, 1, 2).contiguous()
    out_bhds = transpose_bhds_to_bshd(src_bhds, B, H, D, S)
    assert torch.allclose(ref_bhds, out_bhds), "BHDS->BSHD mismatch!"
    print("[OK] Correctness: BHDS -> BSHD")
    print()

    # --- Single transpose benchmark ---
    print("-" * 60)
    print("Single transpose [4,8,1024,64] -> [4,1024,8,64]")
    print("-" * 60)

    t_triton = benchmark_kernel(transpose_bhsd_to_bshd, src, B, H, S, D, warmup=50, rep=200)
    bw_triton = nbytes / (t_triton * 1e-3) / 1e9

    t_eager = benchmark_kernel(eager_transpose, src, B, H, S, D, warmup=50, rep=200)
    bw_eager = nbytes / (t_eager * 1e-3) / 1e9

    compiled_fn = torch.compile(
        lambda x: x.permute(0, 2, 1, 3).contiguous(),
        mode="max-autotune"
    )
    _ = compiled_fn(src)
    torch.cuda.synchronize()
    t_inductor = benchmark_kernel(compiled_fn, src, warmup=50, rep=200)
    bw_inductor = nbytes / (t_inductor * 1e-3) / 1e9

    print(f"  Inductor:  {t_inductor*1000:7.1f} us | {bw_inductor:6.0f} GB/s")
    print(f"  Triton:    {t_triton*1000:7.1f} us | {bw_triton:6.0f} GB/s")
    print(f"  Eager:     {t_eager*1000:7.1f} us | {bw_eager:6.0f} GB/s")
    print()
    print(f"  Triton vs Inductor: {t_inductor/t_triton:.2f}x faster")
    print(f"  Triton vs Eager:    {t_eager/t_triton:.2f}x (target: >= 1.0)")
    print(f"  Eager vs Inductor:  {t_inductor/t_eager:.2f}x")

    # --- Full repro: 12 transposes ---
    print()
    print("-" * 60)
    print("Full repro pattern: 12 transposes (6x BHSD + 6x BHDS)")
    print("-" * 60)

    inputs_sd = [torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') for _ in range(6)]
    inputs_ds = [torch.randn(B, H, D, S, dtype=torch.float16, device='cuda') for _ in range(6)]
    total_nbytes = 12 * nbytes

    def triton_all():
        results = []
        for inp in inputs_sd:
            results.append(transpose_bhsd_to_bshd(inp, B, H, S, D))
        for inp in inputs_ds:
            results.append(transpose_bhds_to_bshd(inp, B, H, D, S))
        return results

    def eager_all():
        results = []
        for inp in inputs_sd:
            results.append(inp.permute(0, 2, 1, 3).contiguous())
        for inp in inputs_ds:
            results.append(inp.permute(0, 3, 1, 2).contiguous())
        return results

    # Verify correctness
    t_res = triton_all()
    e_res = eager_all()
    for i, (t, e) in enumerate(zip(t_res, e_res)):
        assert torch.allclose(t, e), f"Full pattern mismatch at {i}"

    t_triton_full = benchmark_kernel(triton_all, warmup=30, rep=100)
    bw_triton_full = total_nbytes / (t_triton_full * 1e-3) / 1e9

    t_eager_full = benchmark_kernel(eager_all, warmup=30, rep=100)
    bw_eager_full = total_nbytes / (t_eager_full * 1e-3) / 1e9

    print(f"  Total traffic: {total_nbytes / 1024**2:.1f} MB")
    print(f"  Triton (12x): {t_triton_full*1000:7.1f} us | {bw_triton_full:.0f} GB/s")
    print(f"  Eager (12x):  {t_eager_full*1000:7.1f} us | {bw_eager_full:.0f} GB/s")
    print(f"  Triton vs Eager: {t_eager_full/t_triton_full:.2f}x")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Inductor bandwidth:  {bw_inductor:.0f} GB/s")
    print(f"  Triton bandwidth:    {bw_triton:.0f} GB/s")
    print(f"  Eager bandwidth:     {bw_eager:.0f} GB/s")
    print(f"  Triton speedup over Inductor: {t_inductor/t_triton:.2f}x")
    print(f"  Remaining gap to Eager:       {t_triton/t_eager:.2f}x slower")


if __name__ == "__main__":
    main()
