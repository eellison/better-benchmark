"""
Oracle kernel for sum_b691b8dad90a (Demucs training backward).

Pattern: Pointwise ops + cat + reduction over dims [0,2].
    Input: arg31_1[64,64,95696], getitem_27[64,64,95696], arg14_1[64,128,95696]
    Output: f32[128] = cat(mul_tensor_3, mul_tensor_2).sum(dim=[0,2])

Computation graph:
    add_tensor = arg31_1 + getitem_27
    slice_0 = arg14_1[:, :64, :]
    slice_1 = arg14_1[:, 64:128, :]
    sig = sigmoid(slice_1)
    sub = 1 - sig
    mul_a = sub * sig              # sigmoid derivative
    mul_b = mul_a * slice_0
    mul_c = mul_b * add_tensor     # -> mul_tensor_2, goes to out[64:128]
    mul_d = sig * add_tensor       # -> mul_tensor_3, goes to out[0:64]
    output = cat([mul_d, mul_c], dim=1).sum(dim=[0,2])

Oracle strategy:
    The cat + sum(dim=[0,2]) is equivalent to two independent sums:
        out[:64]  = (sig * add_tensor).sum(dim=[0,2])
        out[64:]  = ((1-sig)*sig*slice_0*add_tensor).sum(dim=[0,2])

    All elementwise ops + both reductions are fused into a single Triton kernel.
    This avoids materializing any intermediate tensor (especially the [64,128,95696]
    cat result). The kernel reads each input element exactly once.

    Data volume: 3 inputs totaling ~6GB reads, 128 float output.
    This is purely bandwidth-bound.

    Grid: 2D (channel_block, spatial_tile). Each program reduces a tile of
    the (N, spatial) dimensions for a block of channels, accumulating in
    registers. Partial sums are written to a small buffer, then a finalize
    kernel sums across tiles.

    Since arg14_1 has 128 channels that get sliced into two 64-channel halves,
    we treat each half independently. The kernel processes C_BLOCK channels at
    a time from the 64-channel axis.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

REPRO_ID = "sum_b691b8dad90a"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Triton kernel: fused elementwise + dual reduction
#
# Layout: all inputs are [N=64, C=64, S=95696] contiguous (row-major).
# arg14_1 is [N=64, C=128, S=95696] contiguous.
# We process one channel at a time (from the 64-channel dimension),
# iterating over (N, S) = 64 * 95696 = 6,124,544 elements per channel.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_reduce_kernel(
    arg31_ptr,        # [N, 64, S] f32
    getitem27_ptr,    # [N, 64, S] f32
    arg14_ptr,        # [N, 128, S] f32
    partial_d_ptr,    # [64, N_TILES] f32 - partials for out[:64]
    partial_c_ptr,    # [64, N_TILES] f32 - partials for out[64:]
    N: tl.constexpr,
    C64: tl.constexpr,     # = 64
    S: tl.constexpr,       # = 95696
    C128: tl.constexpr,    # = 128
    N_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Each program handles one (channel, tile) pair.

    channel in [0, 64): the channel index for the 64-channel tensors.
    tile in [0, N_TILES): a spatial tile across the N*S flattened dimension.
    """
    c = tl.program_id(0)       # channel index [0, 64)
    tile_id = tl.program_id(1) # tile index

    total = N * S  # elements per channel to reduce
    tile_size = (total + N_TILES - 1) // N_TILES
    tile_start = tile_id * tile_size
    tile_end = tl.minimum(tile_start + tile_size, total)

    # Strides for [N, C, S] layout: element [n, c, s] at n*C*S + c*S + s
    # For arg31/getitem27: stride is C64*S per batch, S per channel
    # For arg14: stride is C128*S per batch, S per channel
    stride_n_64 = C64 * S    # = 64 * 95696
    stride_n_128 = C128 * S  # = 128 * 95696

    acc_d = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # for mul_tensor_3 (out[:64])
    acc_c = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # for mul_tensor_2 (out[64:])

    for block_start in range(tile_start, tile_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        valid = offsets < tile_end

        # Decompose flat offset into (n, s)
        n = offsets // S
        s = offsets % S

        # Load arg31_1[n, c, s] and getitem_27[n, c, s]
        idx_64 = n * stride_n_64 + c * S + s
        a31 = tl.load(arg31_ptr + idx_64, mask=valid, other=0.0)
        g27 = tl.load(getitem27_ptr + idx_64, mask=valid, other=0.0)

        # add_tensor = arg31_1 + getitem_27
        add_val = a31 + g27

        # Load arg14_1[n, c, s] (slice_0, first 64 channels)
        idx_128_lo = n * stride_n_128 + c * S + s
        slice_0 = tl.load(arg14_ptr + idx_128_lo, mask=valid, other=0.0)

        # Load arg14_1[n, c+64, s] (slice_1, second 64 channels)
        idx_128_hi = n * stride_n_128 + (c + C64) * S + s
        slice_1 = tl.load(arg14_ptr + idx_128_hi, mask=valid, other=0.0)

        # sigmoid(slice_1)
        sig = tl.sigmoid(slice_1)

        # mul_tensor_3 = sig * add_tensor (-> out[:64])
        mul_d = sig * add_val

        # mul_tensor_2 = (1 - sig) * sig * slice_0 * add_tensor (-> out[64:])
        sub_val = 1.0 - sig
        mul_a = sub_val * sig
        mul_b = mul_a * slice_0
        mul_c_val = mul_b * add_val

        acc_d += mul_d
        acc_c += mul_c_val

    # Reduce block-level accumulators to scalar
    sum_d = tl.sum(acc_d, axis=0)
    sum_c = tl.sum(acc_c, axis=0)

    # Store partial results
    tl.store(partial_d_ptr + c * N_TILES + tile_id, sum_d)
    tl.store(partial_c_ptr + c * N_TILES + tile_id, sum_c)


@triton.jit
def _finalize_kernel(
    partial_d_ptr,  # [64, N_TILES]
    partial_c_ptr,  # [64, N_TILES]
    out_ptr,        # [128] output
    N_TILES: tl.constexpr,
):
    """Sum partial results across tiles for each channel."""
    c = tl.program_id(0)  # [0, 64)

    offsets = tl.arange(0, N_TILES)
    base = c * N_TILES

    # Sum partials for out[c] (mul_tensor_3 part)
    partials_d = tl.load(partial_d_ptr + base + offsets)
    tl.store(out_ptr + c, tl.sum(partials_d, axis=0))

    # Sum partials for out[c+64] (mul_tensor_2 part)
    partials_c = tl.load(partial_c_ptr + base + offsets)
    tl.store(out_ptr + c + 64, tl.sum(partials_c, axis=0))


# ---------------------------------------------------------------------------
# Oracle entry point
# ---------------------------------------------------------------------------

def oracle_fused(arg31_1, getitem_27, arg14_1):
    """Fused elementwise + dual reduction.

    Args:
        arg31_1: [64, 64, 95696] f32 contiguous
        getitem_27: [64, 64, 95696] f32 contiguous
        arg14_1: [64, 128, 95696] f32 contiguous

    Returns:
        out: [128] f32 = cat([sig*add, (1-sig)*sig*slice0*add], dim=1).sum([0,2])
    """
    N, C64, S = arg31_1.shape
    C128 = arg14_1.shape[1]
    device = arg31_1.device

    assert arg31_1.is_contiguous()
    assert getitem_27.is_contiguous()
    assert arg14_1.is_contiguous()
    assert C64 == 64
    assert C128 == 128

    # Tuning parameters
    # Total elements per channel: 64 * 95696 = 6,124,544
    # Use enough tiles to saturate GPU: 64 channels * N_TILES programs
    N_TILES = 64  # 64 * 64 = 4096 programs - good GPU occupancy
    BLOCK_SIZE = 4096  # elements per iteration in inner loop

    # N_TILES must be power of 2 for the finalize kernel
    partial_d = torch.empty(C64, N_TILES, dtype=torch.float32, device=device)
    partial_c = torch.empty(C64, N_TILES, dtype=torch.float32, device=device)

    _fused_reduce_kernel[(C64, N_TILES)](
        arg31_1, getitem_27, arg14_1,
        partial_d, partial_c,
        N=N, C64=C64, S=S, C128=C128,
        N_TILES=N_TILES,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    out = torch.empty(128, dtype=torch.float32, device=device)

    _finalize_kernel[(C64,)](
        partial_d, partial_c, out,
        N_TILES=N_TILES,
    )

    return out


# ---------------------------------------------------------------------------
# Reference: eager PyTorch (matches repro.py exactly)
# ---------------------------------------------------------------------------

def reference_pytorch(arg31_1, getitem_27, arg14_1):
    """Direct PyTorch implementation matching repro.py forward()."""
    add_tensor = arg31_1 + getitem_27
    slice_tensor = arg14_1[:, :64, :]
    slice_tensor_1 = arg14_1[:, 64:128, :]
    sigmoid_default = torch.sigmoid(slice_tensor_1)
    sub_tensor = 1.0 - sigmoid_default
    mul_tensor = sub_tensor * sigmoid_default
    mul_tensor_1 = mul_tensor * slice_tensor
    mul_tensor_2 = mul_tensor_1 * add_tensor
    mul_tensor_3 = sigmoid_default * add_tensor
    cat_default = torch.cat([mul_tensor_3, mul_tensor_2], dim=1)
    sum_dim_int_list = cat_default.sum(dim=[0, 2])
    return sum_dim_int_list


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def make_inputs(device: torch.device = None):
    """Generate inputs from the repro module."""
    module = _load_repro_module()
    inputs = module.make_inputs()
    if device is not None:
        moved = []
        for value in inputs:
            if isinstance(value, torch.Tensor):
                moved.append(value.to(device=device))
            else:
                moved.append(value)
        return tuple(moved)
    return tuple(inputs)


def check_correctness(device: torch.device, rtol: float = 1e-4, atol: float = 1e-3):
    """Compare oracle output against eager reference."""
    inputs = make_inputs(device)

    with torch.no_grad():
        oracle_out = oracle_fused(*inputs)
        ref_out = reference_pytorch(*inputs)

    max_diff = (oracle_out.float() - ref_out.float()).abs().max().item()
    rel_diff = ((oracle_out.float() - ref_out.float()).abs() / (ref_out.float().abs() + 1e-8)).max().item()
    close = torch.allclose(oracle_out.float(), ref_out.float(), rtol=rtol, atol=atol)
    print(f"  output: shape={list(oracle_out.shape)}, max_abs_diff={max_diff:.6g}, "
          f"max_rel_diff={rel_diff:.6g}, allclose={close}")

    print(f"  OVERALL: {'PASS' if close else 'FAIL'}")
    return close


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_oracle(device: torch.device, warmup: int = 25, rep: int = 100):
    """Benchmark oracle vs torch.compile + coordinate descent."""
    inputs = make_inputs(device)
    module = _load_repro_module()

    with torch.no_grad():
        # Warmup oracle
        for _ in range(3):
            oracle_fused(*inputs)
        torch.cuda.synchronize()

        # Benchmark oracle
        oracle_ms = triton.testing.do_bench(
            lambda: oracle_fused(*inputs),
            warmup=warmup,
            rep=rep,
        )
        oracle_us = oracle_ms * 1000.0

        # Benchmark torch.compile
        import torch._inductor.config as inductor_config
        model = module.Repro().cuda()
        torch._dynamo.reset()
        compiled = torch.compile(model)
        compiled(*inputs)
        torch.cuda.synchronize()

        compile_ms = triton.testing.do_bench(
            lambda: compiled(*inputs),
            warmup=warmup,
            rep=rep,
        )
        compile_us = compile_ms * 1000.0

        # Benchmark torch.compile + coordinate descent
        torch._dynamo.reset()
        inductor_config.coordinate_descent_tuning = True
        compiled_cd = torch.compile(model)
        compiled_cd(*inputs)
        torch.cuda.synchronize()

        compile_cd_ms = triton.testing.do_bench(
            lambda: compiled_cd(*inputs),
            warmup=warmup,
            rep=rep,
        )
        compile_cd_us = compile_cd_ms * 1000.0
        inductor_config.coordinate_descent_tuning = False

        # Memory bandwidth calculation
        # Reads: arg31_1 (64*64*95696*4) + getitem_27 (same) + arg14_1 (64*128*95696*4)
        bytes_read = (64 * 64 * 95696 * 4) * 2 + (64 * 128 * 95696 * 4)
        bytes_written = 128 * 4  # negligible
        total_bytes = bytes_read + bytes_written
        bw_achieved_gb_s = total_bytes / (oracle_ms * 1e-3) / 1e9

        print(f"\n{'='*60}")
        print(f"Benchmark results for {REPRO_ID}")
        print(f"{'='*60}")
        print(f"  Oracle (fused reduce):                {oracle_us:.1f} us")
        print(f"  torch.compile:                        {compile_us:.1f} us")
        print(f"  torch.compile + coord_descent:        {compile_cd_us:.1f} us")
        print(f"  Speedup (oracle vs compile):          {compile_us/oracle_us:.2f}x")
        print(f"  Speedup (oracle vs coord_descent):    {compile_cd_us/oracle_us:.2f}x")
        print(f"  Achieved bandwidth:                   {bw_achieved_gb_s:.1f} GB/s")
        print(f"  Data volume:                          {total_bytes/1e9:.2f} GB")
        print(f"{'='*60}")
        print(f"\nNote: Oracle fuses all elementwise ops (add, sigmoid, sub, mul)")
        print(f"and both reductions into a single kernel pass, avoiding the")
        print(f"materialization of the [64,128,95696] cat intermediate.")

    return oracle_us, compile_us, compile_cd_us


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if not args.check and not args.bench:
        args.check = True

    if args.check:
        print(f"Correctness check ({REPRO_ID}):")
        check_correctness(device, rtol=args.rtol, atol=args.atol)

    if args.bench:
        print(f"Benchmark ({REPRO_ID}):")
        benchmark_oracle(device, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
