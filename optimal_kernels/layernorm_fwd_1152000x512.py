"""
Hand-optimized Triton kernel for LayerNorm Forward [1152000, 512] bf16.

Pattern:
  x_f32 = x.float()
  var, mean = var_mean(x_f32, dim=-1, correction=0, keepdim=True)
  x_norm = (x_f32 - mean) * rsqrt(var + eps)
  out = (x_norm * weight).to(bf16)

Shape: [1152000, 512] bf16 input, [512] f32 weight
Key: rnumel=512 -> persistent reduction (same as RMSNorm)

On B200, inductor is already at SOL (0.99x gap). This is a pure bandwidth-bound kernel
with small reduction dim that fits entirely in registers.

Inductor uses a tiled persistent kernel [XBLOCK, 512] which processes multiple rows
per program, achieving perfect bandwidth utilization. The `triton_per_fused_*` template
is the correct choice here.

Result: No optimization opportunity on B200. On H100 (~2 TB/s BW), the same kernel
would take ~1180us, and any gap would be from occupancy or launch overhead, not
algorithmic issues.
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def layernorm_fwd_tiled(
    input_ptr,    # bf16[M, N]
    weight_ptr,   # f32[N]
    output_ptr,   # bf16[M, N]
    M,
    N: tl.constexpr,
    eps: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    """Multi-row persistent LayerNorm (matches inductor's approach)."""
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < M
    r_index = tl.arange(0, RBLOCK)[None, :]

    # Load input [XBLOCK, RBLOCK]
    x = tl.load(input_ptr + r_index + N * xindex, xmask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + r_index)

    # Mean
    mean = tl.sum(x, 1)[:, None] / N

    # Variance
    diff = x - mean
    var = tl.sum(diff * diff, 1)[:, None] / N

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    out = diff * rstd * w
    tl.store(output_ptr + r_index + N * xindex, out.to(tl.bfloat16), xmask)


def benchmark_layernorm():
    M, N = 1152000, 512
    eps = 1e-6

    x = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(N, dtype=torch.float32, device='cuda')
    out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

    # Verify correctness
    x_f32 = x.float()
    var, mean = torch.var_mean(x_f32, dim=-1, correction=0, keepdim=True)
    ref = ((x_f32 - mean) * torch.rsqrt(var + eps) * weight).to(torch.bfloat16)

    XBLOCK = 8
    grid = ((M + XBLOCK - 1) // XBLOCK,)
    layernorm_fwd_tiled[grid](x, weight, out, M, N, eps, XBLOCK=XBLOCK, RBLOCK=512, num_warps=8)
    torch.cuda.synchronize()
    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"Max diff (tiled XBLOCK=8): {max_diff:.6f}")

    # Benchmark configs
    print(f"\n{'XBLOCK':>7} {'warps':>6} {'time_us':>10}")
    print("-" * 30)

    best_time = float('inf')
    best_cfg = None

    for xblock in [2, 4, 8, 16]:
        for num_warps in [4, 8, 16]:
            grid = ((M + xblock - 1) // xblock,)
            fn = lambda xb=xblock, nw=num_warps: layernorm_fwd_tiled[((M + xb - 1) // xb,)](
                x, weight, out, M, N, eps, XBLOCK=xb, RBLOCK=512, num_warps=nw
            )
            try:
                ms = do_bench(fn, warmup=25, rep=100, return_mode='min')
                us = ms * 1000
                print(f"{xblock:>7} {num_warps:>6} {us:>10.1f}")
                if us < best_time:
                    best_time = us
                    best_cfg = (xblock, num_warps)
            except Exception as e:
                print(f"{xblock:>7} {num_warps:>6} {'FAILED':>10}")

    # Inductor baseline
    print("\n--- Inductor baseline ---")
    import sys
    sys.path.insert(0, '/tmp/scratch_space/better_benchmark')
    from repros.canonical.var_mean_819a2cfa3958.repro import Repro, make_inputs
    mod = Repro()
    inputs = make_inputs()
    torch._dynamo.reset()
    compiled = torch.compile(mod)
    with torch.no_grad():
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        compiled(*inputs)
    torch.cuda.synchronize()
    inductor_ms = do_bench(lambda: g.replay(), warmup=25, rep=100, return_mode='min')
    inductor_us = inductor_ms * 1000
    print(f"Inductor: {inductor_us:.1f} us")

    # SOL
    total_bytes = M * N * 2 * 2 + N * 4
    sol_elems = total_bytes // 8
    src = torch.empty(sol_elems, dtype=torch.float32, device='cuda')
    dst = torch.empty_like(src)
    sol_ms = do_bench(lambda: torch.add(src, 1, out=dst), warmup=25, rep=100, return_mode='min')
    sol_us = sol_ms * 1000
    del src, dst

    print(f"\n--- Summary ---")
    print(f"SOL (memcpy):  {sol_us:.1f} us")
    print(f"Inductor:      {inductor_us:.1f} us")
    print(f"Best hand:     {best_time:.1f} us (XBLOCK={best_cfg[0]}, warps={best_cfg[1]})")
    print(f"Inductor/SOL:  {inductor_us/sol_us:.2f}x")
    print(f"Hand/SOL:      {best_time/sol_us:.2f}x")
    print(f"NOTE: On B200, inductor already achieves SOL.")


if __name__ == "__main__":
    benchmark_layernorm()
