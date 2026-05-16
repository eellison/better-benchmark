"""
Optimal Triton kernel for channels-last BatchNorm + ReLU.

Target shape: [8, 64, 112, 112] channels-last (strides [802816, 1, 7168, 64])
This is the first BN+ReLU in ResNet50 after the initial convolution.

Key optimization: Uses a 2D grid to avoid modular indexing (integer division).
- axis 0: spatial blocks (tiles over H*W dimension)
- axis 1: batch (N dimension)
- C=64 is small enough to process in a single tile per block

Inductor's approach: flattens to 1D, uses `xindex % C` which requires expensive
integer division on every element. Our approach computes offsets directly from
the 2D grid program_ids.
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def _bn_relu_channels_last_kernel(
    # Pointers
    X_ptr,       # input:  [N, C, H, W] channels-last => strides [H*W*C, 1, W*C, C]
    OUT_ptr,     # output: same layout
    MEAN_ptr,    # running_mean: [C]
    VAR_ptr,     # running_var:  [C]
    WEIGHT_ptr,  # gamma:  [C]
    BIAS_ptr,    # beta:   [C]
    # Shapes
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    # Strides (elements, not bytes)
    stride_n: tl.constexpr,   # H*W*C = 802816
    stride_hw: tl.constexpr,  # C = 64 (this is the W*C / spatial stride)
    # Params
    eps: tl.constexpr,
    # Tile sizes
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    2D grid kernel for BN+ReLU on channels-last tensors.

    Grid: (cdiv(HW, BLOCK_HW), N)
    Each program processes a [BLOCK_HW, BLOCK_C] tile.

    For channels-last [N, C, H, W]:
      - Physical layout is [N, H, W, C] in memory
      - stride_n = H*W*C (batch stride)
      - stride_hw = C (spatial stride — moving one spatial position moves C elements)
      - stride_c = 1 (channel stride — contiguous!)

    Memory access pattern:
      Each thread block reads a contiguous chunk: for a given (n, hw_block),
      it reads BLOCK_HW * C consecutive floats. This is fully coalesced because
      C is the innermost (stride-1) dimension.
    """
    # Program IDs: spatial tile and batch index
    pid_hw = tl.program_id(0)  # which spatial tile
    pid_n = tl.program_id(1)   # which batch element

    # Spatial offsets for this tile
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
    hw_mask = hw_offsets < HW

    # Channel offsets (full C processed per block)
    c_offsets = tl.arange(0, BLOCK_C)  # [BLOCK_C]
    c_mask = c_offsets < C

    # Compute base offset for this batch element
    base_offset = pid_n * stride_n  # batch offset

    # 2D offset computation: [BLOCK_HW, BLOCK_C]
    # offset[i, j] = base_offset + hw_offsets[i] * stride_hw + c_offsets[j] * 1
    offsets = base_offset + hw_offsets[:, None] * stride_hw + c_offsets[None, :]

    # Combined mask: [BLOCK_HW, BLOCK_C]
    mask = hw_mask[:, None] & c_mask[None, :]

    # Load input tile — coalesced since C is stride-1 (innermost)
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # Load BN parameters (broadcast over spatial dim)
    mean = tl.load(MEAN_ptr + c_offsets, mask=c_mask)    # [BLOCK_C]
    var = tl.load(VAR_ptr + c_offsets, mask=c_mask)      # [BLOCK_C]
    weight = tl.load(WEIGHT_ptr + c_offsets, mask=c_mask)  # [BLOCK_C]
    bias = tl.load(BIAS_ptr + c_offsets, mask=c_mask)    # [BLOCK_C]

    # BN inference: (x - mean) * rsqrt(var + eps) * weight + bias
    inv_std = tl.rsqrt(var[None, :] + eps)
    out = (x - mean[None, :]) * inv_std * weight[None, :] + bias[None, :]

    # ReLU
    out = tl.maximum(out, 0.0)

    # Store output — same layout, fully coalesced
    tl.store(OUT_ptr + offsets, out, mask=mask)


def bn_relu_channels_last(x, running_mean, running_var, weight, bias, eps=1e-5):
    """
    Fused BatchNorm inference + ReLU for channels-last tensors.

    Args:
        x: [N, C, H, W] tensor in channels-last format (stride-1 on C)
        running_mean: [C]
        running_var: [C]
        weight: [C] (gamma)
        bias: [C] (beta)
    """
    assert x.is_contiguous(memory_format=torch.channels_last), \
        "Input must be channels-last"

    N, C, H, W = x.shape
    HW = H * W

    # Output in same channels-last layout
    out = torch.empty_like(x)

    # Strides in elements
    stride_n = x.stride(0)  # H*W*C
    stride_hw = x.stride(2)  # W*C ... wait, for channels-last:
    # strides are [H*W*C, 1, W*C, C]
    # stride(0) = H*W*C = 802816
    # stride(1) = 1 (channel stride)
    # stride(2) = W*C = 7168 (height stride)
    # stride(3) = C = 64 (width stride)
    # Physical layout is (N, H, W, C), so spatial stride = C for width, W*C for height

    # We want to iterate over the H*W spatial positions.
    # In channels-last, moving along W (innermost spatial) has stride C=64,
    # and the C dimension is stride 1.
    # We linearize H and W: position (h, w) = h*W + w has memory offset = h*W*C + w*C = (h*W + w)*C
    # So the linear spatial stride is C (every spatial position is C elements apart).
    # This means stride_hw = C for linear spatial indexing!
    stride_hw_val = C  # = x.stride(3) = 64

    # Tile sizes (BLOCK_HW=32 gives best occupancy/throughput balance)
    BLOCK_C = triton.next_power_of_2(C)  # 64
    BLOCK_HW = 32  # 32 spatial positions per block (32 * 64 = 2048 elements per block)

    # Grid: (spatial_tiles, batch_size)
    grid = (triton.cdiv(HW, BLOCK_HW), N)

    _bn_relu_channels_last_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        N=N, C=C, HW=HW,
        stride_n=stride_n,
        stride_hw=stride_hw_val,
        eps=eps,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
    )
    return out


def inductor_bn_relu(x, running_mean, running_var, weight, bias, eps=1e-5):
    """
    Reference: what torch.compile generates (via inductor).
    Uses the same ops as the repro to trigger inductor's pointwise codegen.
    """
    # BN inference
    inv_std = torch.rsqrt(running_var + eps)
    out = (x - running_mean.view(1, -1, 1, 1)) * inv_std.view(1, -1, 1, 1) * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
    # ReLU
    out = torch.relu(out)
    return out


@torch.compile(mode="max-autotune")
def compiled_bn_relu(x, running_mean, running_var, weight, bias, eps=1e-5):
    """torch.compile version — triggers inductor's 1D kernel with mod indexing."""
    inv_std = torch.rsqrt(running_var + eps)
    out = (x - running_mean.view(1, -1, 1, 1)) * inv_std.view(1, -1, 1, 1) * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
    out = torch.relu(out)
    return out


def make_inputs(device='cuda'):
    """Create channels-last inputs matching ResNet50 first BN layer."""
    N, C, H, W = 8, 64, 112, 112

    # Create channels-last tensor (physical layout NHWC)
    x = torch.randn(N, C, H, W, device=device).to(memory_format=torch.channels_last)
    assert x.stride() == (802816, 1, 7168, 64), f"Unexpected strides: {x.stride()}"

    running_mean = torch.randn(C, device=device)
    running_var = torch.rand(C, device=device).abs() + 0.1  # Positive variance
    weight = torch.randn(C, device=device)
    bias = torch.randn(C, device=device)

    return x, running_mean, running_var, weight, bias


def verify_correctness():
    """Verify our kernel matches the reference."""
    x, mean, var, weight, bias = make_inputs()

    # Reference (eager)
    ref = inductor_bn_relu(x, mean, var, weight, bias)

    # Our kernel
    ours = bn_relu_channels_last(x, mean, var, weight, bias)

    # Check
    max_diff = (ref - ours).abs().max().item()
    print(f"Max absolute difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Correctness check failed! max_diff={max_diff}"
    print("Correctness: PASSED")


def benchmark(fn, args, warmup=25, rep=100):
    """Benchmark using triton's timing utility."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(rep):
        start_event.record()
        fn(*args)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times = sorted(times)
    # Use median
    median_ms = times[len(times) // 2]
    return median_ms


def main():
    print("=" * 70)
    print("Channels-Last BatchNorm + ReLU: Optimal 2D-Grid vs Inductor")
    print("=" * 70)
    print(f"\nShape: [8, 64, 112, 112] channels-last")
    print(f"Strides: [802816, 1, 7168, 64]")
    print(f"Tensor size: {8*64*112*112*4 / 1024 / 1024:.1f} MB\n")

    # Verify correctness
    verify_correctness()
    print()

    x, mean, var, weight, bias = make_inputs()

    # --- Benchmark our optimal kernel ---
    print("Benchmarking optimal 2D-grid kernel...")
    optimal_ms = benchmark(bn_relu_channels_last, (x, mean, var, weight, bias))
    print(f"  Optimal (2D grid, no mod):  {optimal_ms:.4f} ms")

    # --- Benchmark inductor (torch.compile) ---
    print("Benchmarking inductor (torch.compile max-autotune)...")
    # Warmup compile
    _ = compiled_bn_relu(x, mean, var, weight, bias)
    torch.cuda.synchronize()
    inductor_ms = benchmark(compiled_bn_relu, (x, mean, var, weight, bias))
    print(f"  Inductor (1D grid, mod idx): {inductor_ms:.4f} ms")

    # --- Bandwidth calculation ---
    # Read input (32MB) + write output (32MB) + read params (4 * 64 * 4B = 1KB, negligible)
    tensor_bytes = 8 * 64 * 112 * 112 * 4
    total_bytes = 2 * tensor_bytes  # read + write
    total_gb = total_bytes / 1e9

    optimal_bw = total_gb / (optimal_ms / 1000)
    inductor_bw = total_gb / (inductor_ms / 1000)

    print(f"\n{'Metric':<30} {'Optimal':>12} {'Inductor':>12} {'Speedup':>10}")
    print("-" * 70)
    print(f"{'Latency (ms)':<30} {optimal_ms:>12.4f} {inductor_ms:>12.4f} {inductor_ms/optimal_ms:>9.2f}x")
    print(f"{'Bandwidth (GB/s)':<30} {optimal_bw:>12.1f} {inductor_bw:>12.1f} {'':>10}")
    print()

    # --- Analysis ---
    print("Analysis:")
    print(f"  - Our kernel: 2D grid ({triton.cdiv(112*112, 32)} spatial tiles x 8 batch)")
    print(f"    No integer division for index decomposition.")
    print(f"    Coalesced reads: each warp reads consecutive C=64 floats (256B lines).")
    print(f"  - Inductor: 1D grid over N*H*W*C = {8*64*112*112} elements")
    print(f"    Uses xindex % 64 for channel index (integer division per element).")
    print()

    if optimal_ms < inductor_ms:
        pct = (1 - optimal_ms / inductor_ms) * 100
        print(f"Result: Optimal kernel is {pct:.1f}% faster than inductor.")
    else:
        pct = (1 - inductor_ms / optimal_ms) * 100
        print(f"Result: Inductor is {pct:.1f}% faster (unexpected — investigate).")

    return optimal_ms, inductor_ms


if __name__ == "__main__":
    main()
