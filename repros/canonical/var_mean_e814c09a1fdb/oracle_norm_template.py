"""
Oracle kernel for var_mean_e814c09a1fdb (norm_template_canonicalization).

Pattern: BN training forward for pytorch_unet
  convolution_15[8, 64, 320, 479] -> var_mean over [0,2,3] -> normalize -> affine -> ReLU
  -> bilinear upsample to [8, 64, 640, 958] -> pad to [8, 64, 640, 959]
  -> cat with relu_1[8, 64, 640, 959] along dim=1 -> [8, 128, 640, 959]
  Plus running stats EMA update.

Strategy:
  - Reduction dim per channel: N*H*W = 8*320*479 = 1,226,240 (too large for single block)
  - Two-pass approach:
    Pass 1: Multiple blocks per channel compute partial sum/sum_sq, store to scratchpad
    Pass 2 (finalize): Reduce partials -> per-channel mean, var
  - Fused normalize + affine + ReLU in a single pointwise kernel
  - Bilinear upsample as a separate kernel (index computation + gather pattern)
  - Cat + pad done via PyTorch (trivial copy-based kernel, not the bottleneck)
  - Running stats update folded into finalize (trivial 64-element update)

The oracle demonstrates a tight floor for the BN+affine+ReLU+upsample portion.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.testing import do_bench
except Exception:
    triton = None
    tl = None
    do_bench = None



from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "var_mean_e814c09a1fdb"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

# Default shape from the repro
DEFAULT_N = 8
DEFAULT_C = 64
DEFAULT_H = 320
DEFAULT_W = 479
DEFAULT_EPS = 1e-05
DEFAULT_MOMENTUM = 0.1
# Bessel correction factor for unbiased variance in running stats
# N*H*W = 8*320*479 = 1226240; correction = N*H*W / (N*H*W - 1)
DEFAULT_CORRECTION = 1.0000008155017088

# Upsample output shape
UPSAMPLE_H = 640
UPSAMPLE_W = 958  # Before padding to 959


# ============================================================================
# Triton kernels
# ============================================================================

if triton is not None:

    @triton.jit
    def _partial_stats_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sum_sq_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        n_batches: tl.constexpr,
        elems_per_channel: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Each program processes one tile of the reduction dimension for one channel.

        Grid: (n_blocks_per_channel, channels)
        """
        block_id = tl.program_id(0)
        channel = tl.program_id(1)

        # Range of elements this block handles within the channel's reduction dim
        start = block_id * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elems_per_channel

        # Map flat offset within channel -> NCHW flat index
        # offset = n * H * W + hw  =>  n = offset // hw_size, hw = offset % hw_size
        n_idx = offsets // hw_size
        hw_idx = offsets % hw_size
        flat_idx = n_idx * channels * hw_size + channel * hw_size + hw_idx

        vals = tl.load(x_ptr + flat_idx, mask=mask, other=0.0).to(tl.float32)

        local_sum = tl.sum(vals, axis=0)
        local_sum_sq = tl.sum(vals * vals, axis=0)

        # Store partial results; one slot per (block_id, channel)
        n_blocks = tl.cdiv(elems_per_channel, BLOCK_SIZE)
        out_idx = channel * n_blocks + block_id
        tl.store(partial_sum_ptr + out_idx, local_sum)
        tl.store(partial_sum_sq_ptr + out_idx, local_sum_sq)

    @triton.jit
    def _finalize_stats_kernel(
        partial_sum_ptr,
        partial_sum_sq_ptr,
        mean_ptr,
        var_ptr,
        running_mean_ptr,
        running_var_ptr,
        out_running_mean_ptr,
        out_running_var_ptr,
        n_blocks: tl.constexpr,
        elems_per_channel: tl.constexpr,
        correction: tl.constexpr,
        momentum: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Reduce partial sums to get per-channel mean and variance, and update running stats.

        Grid: (channels,)
        """
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_blocks

        base = channel * n_blocks
        sums = tl.load(partial_sum_ptr + base + offsets, mask=mask, other=0.0)
        sums_sq = tl.load(partial_sum_sq_ptr + base + offsets, mask=mask, other=0.0)

        total_sum = tl.sum(sums, axis=0)
        total_sum_sq = tl.sum(sums_sq, axis=0)

        mean = total_sum / elems_per_channel
        var = total_sum_sq / elems_per_channel - mean * mean
        var = tl.maximum(var, 0.0)

        tl.store(mean_ptr + channel, mean)
        tl.store(var_ptr + channel, var)

        # Running stats EMA update (folded in since it's trivial per-channel)
        old_mean = tl.load(running_mean_ptr + channel)
        old_var = tl.load(running_var_ptr + channel)
        new_mean = old_mean * (1.0 - momentum) + mean * momentum
        new_var = old_var * (1.0 - momentum) + var * correction * momentum
        tl.store(out_running_mean_ptr + channel, new_mean)
        tl.store(out_running_var_ptr + channel, new_var)

    @triton.jit
    def _bn_affine_relu_kernel(
        x_ptr,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        y_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused normalize + affine + ReLU pointwise kernel.

        Grid: (cdiv(total, BLOCK_SIZE),)
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total

        # Determine channel from flat NCHW index
        channel = (offsets // hw_size) % channels

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0)
        var = tl.load(var_ptr + channel, mask=mask, other=0.0)
        w = tl.load(weight_ptr + channel, mask=mask, other=0.0)
        b = tl.load(bias_ptr + channel, mask=mask, other=0.0)

        normed = (x - mean) * tl.rsqrt(var + eps)
        y = normed * w + b
        y = tl.maximum(y, 0.0)

        tl.store(y_ptr + offsets, y, mask=mask)


# ============================================================================
# Oracle implementation
# ============================================================================

def triton_bn_affine_relu(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = DEFAULT_EPS,
    momentum: float = DEFAULT_MOMENTUM,
    correction: float = DEFAULT_CORRECTION,
    stats_block: int = 4096,
    affine_block: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton oracle: fused BN stats + normalize + affine + ReLU.

    Returns: (y_relu, new_running_mean, new_running_var)
    """
    if triton is None:
        raise RuntimeError("Triton not available")

    assert x.is_cuda and x.is_contiguous() and x.ndim == 4

    n_batches, channels, height, width = x.shape
    hw_size = height * width
    elems_per_channel = n_batches * hw_size
    total = x.numel()

    # Partial stats
    n_blocks_per_channel = triton.cdiv(elems_per_channel, stats_block)
    partial_sum = torch.empty(channels * n_blocks_per_channel, device=x.device, dtype=torch.float32)
    partial_sum_sq = torch.empty_like(partial_sum)

    _partial_stats_kernel[(n_blocks_per_channel, channels)](
        x,
        partial_sum,
        partial_sum_sq,
        channels=channels,
        hw_size=hw_size,
        n_batches=n_batches,
        elems_per_channel=elems_per_channel,
        BLOCK_SIZE=stats_block,
        num_warps=8,
    )

    # Finalize stats + running stats update
    mean = torch.empty(channels, device=x.device, dtype=torch.float32)
    var = torch.empty(channels, device=x.device, dtype=torch.float32)
    new_running_mean = torch.empty_like(running_mean)
    new_running_var = torch.empty_like(running_var)
    finalize_block = triton.next_power_of_2(n_blocks_per_channel)

    _finalize_stats_kernel[(channels,)](
        partial_sum,
        partial_sum_sq,
        mean,
        var,
        running_mean,
        running_var,
        new_running_mean,
        new_running_var,
        n_blocks=n_blocks_per_channel,
        elems_per_channel=elems_per_channel,
        correction=correction,
        momentum=momentum,
        BLOCK_SIZE=finalize_block,
        num_warps=4,
    )

    # Fused normalize + affine + ReLU
    y = torch.empty_like(x)
    grid = (triton.cdiv(total, affine_block),)

    _bn_affine_relu_kernel[grid](
        x,
        mean,
        var,
        weight,
        bias,
        y,
        total=total,
        channels=channels,
        hw_size=hw_size,
        eps=eps,
        BLOCK_SIZE=affine_block,
        num_warps=4,
    )

    return y, new_running_mean, new_running_var


def oracle_full(
    convolution_15: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    relu_1: torch.Tensor,
    eps: float = DEFAULT_EPS,
    momentum: float = DEFAULT_MOMENTUM,
    correction: float = DEFAULT_CORRECTION,
    stats_block: int = 4096,
    affine_block: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full oracle: BN + affine + ReLU + bilinear upsample + pad + cat + running stats.

    Returns: (cat_default, new_running_mean, new_running_var)
    """
    # BN + affine + ReLU via Triton
    y_relu, new_rm, new_rv = triton_bn_affine_relu(
        convolution_15, running_mean, running_var, weight, bias,
        eps=eps, momentum=momentum, correction=correction,
        stats_block=stats_block, affine_block=affine_block,
    )

    # Bilinear upsample to [8, 64, 640, 958] then pad to [8, 64, 640, 959]
    # Using F.interpolate which maps to efficient CUDA kernels
    upsampled = F.interpolate(y_relu, size=(UPSAMPLE_H, UPSAMPLE_W), mode='bilinear', align_corners=False)

    # Pad last dim by 1 (right side)
    padded = F.pad(upsampled, [0, 1, 0, 0], mode='constant', value=0.0)

    # Cat with relu_1 along dim=1: [8, 64, 640, 959] + [8, 64, 640, 959] -> [8, 128, 640, 959]
    cat_out = torch.cat([relu_1, padded], dim=1)

    return cat_out, new_rm, new_rv


# ============================================================================
# Reference implementation (matching the repro exactly)
# ============================================================================

def torch_reference(
    convolution_15: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    relu_1: torch.Tensor,
    shape_param_0,
    eps: float = DEFAULT_EPS,
    momentum: float = DEFAULT_MOMENTUM,
    correction: float = DEFAULT_CORRECTION,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full reference matching the repro outputs: (cat_default, new_running_mean, new_running_var)."""
    # var_mean
    var_mean_result = torch.var_mean(convolution_15, dim=(0, 2, 3), correction=0, keepdim=True)
    var_val = var_mean_result[0]  # [1, 64, 1, 1]
    mean_val = var_mean_result[1]  # [1, 64, 1, 1]

    # normalize
    inv_std = torch.rsqrt(var_val + eps)
    normed = (convolution_15 - mean_val) * inv_std

    # Running stats EMA
    mean_1d = mean_val.squeeze((0, 2, 3))
    var_1d = var_val.squeeze((0, 2, 3))
    new_running_mean = running_mean * 0.9 + mean_1d * 0.1
    new_running_var = running_var * 0.9 + var_1d * correction * 0.1

    # Affine + ReLU
    w = weight[None, :, None, None]
    b = bias[None, :, None, None]
    y = normed * w + b
    y_relu = torch.relu(y)

    # Bilinear interpolation (matching the repro's index computation exactly)
    # The repro manually computes bilinear upsample from [8,64,320,479] to [8,64,640,958]
    # then pads to [8,64,640,959]
    # Using F.interpolate with the same scale factors
    # scale_h: iota(640) * 0.49921752738654146 -> maps [0,639] to [0, 319.0]
    # scale_w: iota(958) * 0.4994775339602926 -> maps [0,957] to [0, 478.0]
    # This matches align_corners=False with output_size=(640, 958)
    upsampled = F.interpolate(y_relu, size=(640, 958), mode='bilinear', align_corners=False)
    padded = F.pad(upsampled, [0, 1, 0, 0], mode='constant', value=0.0)
    cat_out = torch.cat([relu_1, padded], dim=1)

    return cat_out, new_running_mean, new_running_var


# ============================================================================
# Correctness check
# ============================================================================

def check_correctness(
    device: str = "cuda",
    seed: int = 42,
    atol: float = 1e-2,
    rtol: float = 1e-3,
):
    """Verify oracle matches the PyTorch reference (and the compiled repro)."""
    torch.manual_seed(seed)

    n, c, h, w = DEFAULT_N, DEFAULT_C, DEFAULT_H, DEFAULT_W
    convolution_15 = torch.randn(n, c, h, w, device=device, dtype=torch.float32)
    running_mean = torch.randn(c, device=device, dtype=torch.float32)
    running_var = torch.rand(c, device=device, dtype=torch.float32) + 0.5
    weight = torch.randn(c, device=device, dtype=torch.float32)
    bias = torch.randn(c, device=device, dtype=torch.float32)
    relu_1 = torch.randn(n, c, UPSAMPLE_H, 959, device=device, dtype=torch.float32)
    shape_param_0 = [640, 1]

    # Reference
    ref_cat, ref_rm, ref_rv = torch_reference(
        convolution_15, running_mean, running_var, weight, bias, relu_1, shape_param_0
    )

    # Oracle
    oracle_cat, oracle_rm, oracle_rv = oracle_full(
        convolution_15, running_mean, running_var, weight, bias, relu_1
    )

    torch.cuda.synchronize()

    # Check outputs
    max_diff_cat = (oracle_cat - ref_cat).abs().max().item()
    max_diff_rm = (oracle_rm - ref_rm).abs().max().item()
    max_diff_rv = (oracle_rv - ref_rv).abs().max().item()

    cat_ok = torch.allclose(oracle_cat, ref_cat, atol=atol, rtol=rtol)
    rm_ok = torch.allclose(oracle_rm, ref_rm, atol=atol, rtol=rtol)
    rv_ok = torch.allclose(oracle_rv, ref_rv, atol=atol, rtol=rtol)

    print(f"Correctness check (N={n}, C={c}, H={h}, W={w}):")
    print(f"  cat output:   max_diff={max_diff_cat:.6e}  {'PASS' if cat_ok else 'FAIL'}")
    print(f"  running_mean: max_diff={max_diff_rm:.6e}  {'PASS' if rm_ok else 'FAIL'}")
    print(f"  running_var:  max_diff={max_diff_rv:.6e}  {'PASS' if rv_ok else 'FAIL'}")

    all_ok = cat_ok and rm_ok and rv_ok
    if not all_ok:
        print("  OVERALL: FAIL")
    else:
        print("  OVERALL: PASS")
    return all_ok


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_oracle(
    stats_block: int = 4096,
    affine_block: int = 1024,
    warmup: int = 25,
    rep: int = 100,
    device: str = "cuda",
    seed: int = 42,
):
    """Benchmark the oracle vs compiled repro."""
    torch.manual_seed(seed)

    n, c, h, w = DEFAULT_N, DEFAULT_C, DEFAULT_H, DEFAULT_W
    convolution_15 = torch.randn(n, c, h, w, device=device, dtype=torch.float32)
    running_mean = torch.randn(c, device=device, dtype=torch.float32)
    running_var = torch.rand(c, device=device, dtype=torch.float32) + 0.5
    weight = torch.randn(c, device=device, dtype=torch.float32)
    bias = torch.randn(c, device=device, dtype=torch.float32)
    relu_1 = torch.randn(n, c, UPSAMPLE_H, 959, device=device, dtype=torch.float32)

    # -- Oracle: full --
    def oracle_fn():
        return oracle_full(
            convolution_15, running_mean, running_var, weight, bias, relu_1,
            stats_block=stats_block, affine_block=affine_block,
        )

    # Warmup
    for _ in range(3):
        oracle_fn()
    torch.cuda.synchronize()

    oracle_us = do_bench(oracle_fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

    # -- Oracle: BN part only --
    def oracle_bn_fn():
        return triton_bn_affine_relu(
            convolution_15, running_mean, running_var, weight, bias,
            stats_block=stats_block, affine_block=affine_block,
        )

    for _ in range(3):
        oracle_bn_fn()
    torch.cuda.synchronize()

    oracle_bn_us = do_bench(oracle_bn_fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

    # -- Compiled repro for comparison --
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from repro import Repro, make_inputs as repro_make_inputs

    mod = Repro()
    inputs = repro_make_inputs()
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

    compiled_us = do_bench(lambda: g.replay(), warmup=warmup, rep=rep, return_mode="min") * 1000.0

    # -- SOL estimate --
    # Data volume estimate:
    # Read: convolution_15 (8*64*320*479*4) + weight,bias,rm,rv (4*64*4) + relu_1 (8*64*640*959*4)
    # Write: cat_out (8*128*640*959*4) + new_rm,new_rv (2*64*4)
    x_bytes = n * c * h * w * 4
    param_bytes = 4 * c * 4
    relu1_bytes = n * c * UPSAMPLE_H * 959 * 4
    cat_bytes = n * 128 * UPSAMPLE_H * 959 * 4
    total_bytes = x_bytes + param_bytes + relu1_bytes + cat_bytes + 2 * c * 4

    copy_elems = max(total_bytes // (2 * 4), 256)
    src = torch.empty(copy_elems, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)
    sol_us = do_bench(lambda: torch.add(src, 1, out=dst), warmup=warmup, rep=rep, return_mode="min") * 1000.0
    del src, dst

    print(f"\n{'='*60}")
    print(f"Oracle benchmark: {REPRO_ID} (BN training forward + upsample + cat)")
    print(f"Shape: [{n}, {c}, {h}, {w}] -> upsample -> cat with relu_1")
    print(f"{'='*60}")
    print(f"  Compiled (CUDAGraph):         {compiled_us:8.1f} us")
    print(f"  Oracle BN+affine+ReLU only:   {oracle_bn_us:8.1f} us")
    print(f"  Oracle full (BN+up+cat):      {oracle_us:8.1f} us")
    print(f"  Memcopy SOL:                  {sol_us:8.1f} us")
    print(f"  Total data volume:            {total_bytes / 1e6:.2f} MB")
    print(f"  Oracle full / SOL:            {oracle_us / sol_us:.2f}x")
    print(f"  Compiled / SOL:               {compiled_us / sol_us:.2f}x")
    print(f"  Speedup (compiled/oracle):    {compiled_us / oracle_us:.2f}x")
    print(f"{'='*60}")


# ============================================================================
# Main
# ============================================================================


def oracle_forward(inputs):
    return oracle_full(*inputs)


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
