"""
Benchmark: fused sigmoid+add+top2 Triton kernel vs ATen topk fallback.
"""
import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def fused_sigmoid_add_top2_kernel(
    logits_ptr,     # [2048, 256] f32
    bias_ptr,       # [256] bf16
    out_ptr,        # [2048*8*2] f32 (values only, contiguous)
    n_rows: tl.constexpr,    # 2048*8 = 16384
    DIM: tl.constexpr,       # 32
    BLOCK_M: tl.constexpr,   # rows per block
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M

    for row_offset in range(BLOCK_M):
        row = row_start + row_offset
        if row < n_rows:
            # Global offset in the flat [2048, 256] tensor
            base_offset = row * DIM

            # Load all 32 elements
            offsets = tl.arange(0, DIM)
            vals = tl.load(logits_ptr + base_offset + offsets)

            # Apply sigmoid
            vals = tl.sigmoid(vals)

            # Add bias (periodic with period 256)
            group_in_row = row % 8
            bias_offsets = group_in_row * DIM + offsets
            bias_vals = tl.load(bias_ptr + bias_offsets).to(tl.float32)
            vals = vals + bias_vals

            # Find top-2 with simple register-level approach for DIM=32
            max1 = tl.max(vals, axis=0)
            # Mask out max1 positions
            is_max1 = (vals == max1)
            vals2 = tl.where(is_max1, float('-inf'), vals)
            max2 = tl.max(vals2, axis=0)

            # Store top2 values
            out_row = row * 2
            tl.store(out_ptr + out_row, max1)
            tl.store(out_ptr + out_row + 1, max2)


@triton.jit
def topk2_only_kernel(
    input_ptr,      # [16384, 32] f32
    out_ptr,        # [16384, 2] f32
    n_rows: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """TopK=2 only (no sigmoid/add), for isolating topk cost."""
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M

    for row_offset in range(BLOCK_M):
        row = row_start + row_offset
        if row < n_rows:
            base_offset = row * DIM
            offsets = tl.arange(0, DIM)
            vals = tl.load(input_ptr + base_offset + offsets)

            max1 = tl.max(vals, axis=0)
            is_max1 = (vals == max1)
            vals2 = tl.where(is_max1, float('-inf'), vals)
            max2 = tl.max(vals2, axis=0)

            out_row = row * 2
            tl.store(out_ptr + out_row, max1)
            tl.store(out_ptr + out_row + 1, max2)


def main():
    torch.manual_seed(42)
    logits = torch.randn(2048, 256, device='cuda', dtype=torch.float32)
    bias = torch.randn(256, device='cuda', dtype=torch.bfloat16)
    out = torch.empty(2048, 8, 2, device='cuda', dtype=torch.float32)

    n_rows = 2048 * 8  # 16384
    DIM = 32

    # Try different BLOCK_M values
    for BLOCK_M in [1, 2, 4, 8, 16]:
        grid = (n_rows // BLOCK_M,)
        # Warmup
        fused_sigmoid_add_top2_kernel[grid](logits, bias, out, n_rows, DIM, BLOCK_M)
        torch.cuda.synchronize()

        # Verify correctness
        ref = (logits.sigmoid() + bias.unsqueeze(0).float()).view(2048, 8, 32).topk(2, dim=-1)[0]
        max_diff = (out - ref).abs().max().item()

        ms = do_bench(lambda: fused_sigmoid_add_top2_kernel[grid](logits, bias, out, n_rows, DIM, BLOCK_M), warmup=50, rep=200)
        print(f'Fused Triton (BLOCK_M={BLOCK_M:2d}): {ms*1000:.1f} us  (max_diff={max_diff:.6f})')

    # TopK-only Triton kernel
    print("\n--- TopK-only comparison ---")
    topk_input = (logits.sigmoid() + bias.unsqueeze(0).float()).view(2048*8, 32).contiguous()
    topk_out = torch.empty(2048*8, 2, device='cuda', dtype=torch.float32)

    for BLOCK_M in [1, 2, 4, 8, 16]:
        grid = (n_rows // BLOCK_M,)
        topk2_only_kernel[grid](topk_input, topk_out, n_rows, DIM, BLOCK_M)
        torch.cuda.synchronize()

        ms = do_bench(lambda: topk2_only_kernel[grid](topk_input, topk_out, n_rows, DIM, BLOCK_M), warmup=50, rep=200)
        print(f'Triton topk2 only (BLOCK_M={BLOCK_M:2d}): {ms*1000:.1f} us')

    # ATen topk (baseline)
    print("\n--- ATen baselines ---")
    topk_input_3d = topk_input.view(2048, 8, 32)
    topk_ms = do_bench(lambda: torch.topk(topk_input_3d, 2, dim=-1), warmup=50, rep=200)
    print(f'ATen topk [2048,8,32] k=2 dim=-1: {topk_ms*1000:.1f} us')

    # Compiled full pipeline
    class Repro(torch.nn.Module):
        def forward(self, mm, arg2_1):
            sigmoid_default = torch.ops.aten.sigmoid.default(mm)
            unsqueeze_default = torch.ops.aten.unsqueeze.default(arg2_1, 0)
            add_tensor = torch.ops.aten.add.Tensor(sigmoid_default, unsqueeze_default)
            reshape_default = torch.ops.aten.reshape.default(add_tensor, [-1, 8, 32])
            topk_default = torch.ops.aten.topk.default(reshape_default, 2)
            return topk_default[0]

    mod = Repro()
    compiled = torch.compile(mod)
    inputs = [logits, bias]
    for _ in range(3):
        compiled(*inputs)
    torch.cuda.synchronize()
    compiled_ms = do_bench(lambda: compiled(*inputs), warmup=50, rep=200)
    print(f'torch.compile (2 kernels: triton pointwise + ATen topk): {compiled_ms*1000:.1f} us')

    # SOL
    total_bytes = 2048*256*4 + 256*2 + 2048*8*2*4
    copy_elems = total_bytes // (2*4)
    src = torch.empty(copy_elems, dtype=torch.float32, device='cuda')
    dst = torch.empty_like(src)
    sol_ms = do_bench(lambda: dst.copy_(src), warmup=50, rep=200)
    sol_us = sol_ms * 1000
    print(f'Memcopy SOL ({total_bytes/1024:.1f} KB): {sol_us:.1f} us')

    print(f'\n=== SUMMARY ===')
    # Best fused
    best_fused_ms = min(do_bench(lambda: fused_sigmoid_add_top2_kernel[(n_rows // bm,)](logits, bias, out, n_rows, DIM, bm), warmup=50, rep=200) for bm in [1, 2, 4, 8, 16])
    print(f'Best fused Triton / SOL:     {best_fused_ms*1000/sol_us:.2f}x')
    print(f'torch.compile / SOL:         {compiled_ms*1000/sol_us:.2f}x')
    print(f'ATen topk alone / SOL:       {topk_ms*1000/sol_us:.2f}x')
    print(f'Speedup (compile -> fused):  {compiled_ms/best_fused_ms:.2f}x')


if __name__ == "__main__":
    main()
