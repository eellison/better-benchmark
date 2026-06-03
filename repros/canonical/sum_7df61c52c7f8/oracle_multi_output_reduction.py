"""
Oracle kernel for sum_7df61c52c7f8 (AlexNet training backward, max pool backward + reduction).

Pattern: scatter_add -> reshape -> where -> sum(dim=[0,2,3])
    Input:
      - getitem_9:  f32[1024, 64, 27, 27]  (source values for max pool backward)
      - arg10_1:    i8[1024, 64, 27, 27]   (max pool offset indices, values in [0, 9))
      - arg25_1:    b8[1024, 64, 55, 55]   (boolean mask for where)
      - full:       f32[]                    (scalar fill value for masked positions)
      - shape params: [65536, 729], [65536, 729], [1024, 64, 55, 55]

    Computation:
      1. full_default = zeros([65536, 3025])
      2. view of getitem_9 -> [65536, 729]
      3. indices = _low_memory_max_pool_offsets_to_indices(arg10_1, kernel=[3,3],
           output_size=[55,55], stride=[2,2], padding=[0,0], dilation=[1,1])
      4. view of indices -> [65536, 729]
      5. scatter_add(full_default, 1, indices_view, values_view) -> [65536, 3025]
      6. reshape -> [1024, 64, 55, 55]
      7. where(arg25_1, full, scatter_result) -> [1024, 64, 55, 55]
      8. sum(dim=[0,2,3]) -> [64]

    Output: f32[64]

Oracle strategy:
    The original graph materializes a large [65536, 3025] = ~800 MB scatter_add
    intermediate tensor. Since the final reduction sums over dims [0, 2, 3]
    (batch and spatial), we can fuse the scatter with the mask check and accumulate
    directly into per-channel partial sums, avoiding the intermediate entirely.

    Key insight: for each source value at (batch, channel, src_h, src_w), the pool
    index tells us which output spatial position it lands at. We check the mask at
    that (batch, channel, out_h, out_w) position; if not masked, we add the source
    value to the channel accumulator.

    The per-channel sum decomposes into:
      sum[c] = full_val * count_where_mask_true[c]
             + sum_of_source_values_at_unmasked_destinations[c]

    Part A covers positions where no source value scatters (they remain zero) or
    where the mask zeroes them out anyway.  Part B only iterates over the
    1024*729 = 745,472 source positions per channel (vs 1024*3025 = 3,097,600
    output positions), which is a significant reduction.

    Implementation uses pure PyTorch with advanced indexing (gather-mask-reduce),
    which avoids the large intermediate while remaining simple and portable.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

REPRO_ID = "sum_7df61c52c7f8"
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
# Triton kernel: fused gather-mask-reduce
#
# For each channel c, we iterate over (batch, src_h, src_w) = 1024 * 27 * 27
# = 746,496 source positions. For each source position, we:
#   1. Compute the destination index using _low_memory_max_pool_offsets_to_indices logic
#   2. Check the mask at the destination
#   3. If not masked, accumulate the source value
#
# Grid: 2D (channel, tile). Partials are summed in a finalize pass.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_scatter_reduce_kernel(
    src_ptr,          # f32[N, C, src_H, src_W] = [1024, 64, 27, 27]
    offsets_ptr,      # i8[N, C, src_H, src_W]
    mask_ptr,         # b8[N, C, out_H, out_W] = [1024, 64, 55, 55]
    partial_ptr,      # f32[C, N_TILES]
    N: tl.constexpr,
    C: tl.constexpr,
    src_H: tl.constexpr,
    src_W: tl.constexpr,
    out_H: tl.constexpr,
    out_W: tl.constexpr,
    stride_h: tl.constexpr,   # pool stride = 2
    stride_w: tl.constexpr,   # pool stride = 2
    kernel_w: tl.constexpr,   # pool kernel width = 3
    total_src_per_channel: tl.constexpr,   # N * src_H * src_W
    N_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Each program processes one (channel, tile) pair, accumulating source values
    whose destination positions are not masked."""
    c = tl.program_id(0)
    tile_id = tl.program_id(1)

    tile_size = (total_src_per_channel + N_TILES - 1) // N_TILES
    tile_start = tile_id * tile_size
    tile_end = tile_start + tile_size
    if tile_end > total_src_per_channel:
        tile_end = total_src_per_channel

    # Strides for [N, C, H, W] layout
    src_stride_n = C * src_H * src_W
    src_stride_c = src_H * src_W
    mask_stride_n = C * out_H * out_W
    mask_stride_c = out_H * out_W

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for block_start in range(tile_start, tile_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        valid = offsets < tile_end

        # Decompose flat offset into (n, src_h, src_w)
        n = offsets // (src_H * src_W)
        rem = offsets % (src_H * src_W)
        sh = rem // src_W
        sw = rem % src_W

        # Load source value
        src_idx = n * src_stride_n + c * src_stride_c + sh * src_W + sw
        src_val = tl.load(src_ptr + src_idx, mask=valid, other=0.0)

        # Load pool offset (i8)
        off_val = tl.load(offsets_ptr + src_idx, mask=valid, other=0).to(tl.int32)

        # Reconstruct destination position from offset
        # _low_memory_max_pool_offsets_to_indices logic for kernel=[3,3], stride=[2,2]:
        # For source position (sh, sw), the output position is at:
        #   out_h = sh * stride_h + off_val // kernel_w
        #   out_w = sw * stride_w + off_val % kernel_w
        out_h = sh * stride_h + off_val // kernel_w
        out_w = sw * stride_w + off_val % kernel_w

        # Load mask at destination (torch.bool is stored as uint8: 0=False, 1=True)
        mask_idx = n * mask_stride_n + c * mask_stride_c + out_h * out_W + out_w
        is_masked = tl.load(mask_ptr + mask_idx, mask=valid, other=1)

        # Accumulate source value only where NOT masked
        # where(mask=True, full_val, scatter_result) -> masked positions get full_val,
        # unmasked positions get the scatter_add result
        # We only want to accumulate the scatter contributions at unmasked positions
        # Use == 0 instead of ~ to avoid bitwise NOT issues with int8 booleans
        not_masked = (is_masked == 0)
        contribute = valid & not_masked
        acc += tl.where(contribute, src_val, 0.0)

    # Reduce block-level accumulator to scalar
    total = tl.sum(acc, axis=0)
    tl.store(partial_ptr + c * N_TILES + tile_id, total)


@triton.jit
def _finalize_kernel(
    partial_ptr,     # f32[C, N_TILES]
    mask_count_ptr,  # f32[C] - count of masked positions per channel
    out_ptr,         # f32[C]
    full_val,        # scalar fill value
    N_TILES: tl.constexpr,
    BLOCK_N_TILES: tl.constexpr,
):
    """Sum partial scatter results and add the mask contribution."""
    c = tl.program_id(0)

    # Sum partials from all tiles
    offs = tl.arange(0, BLOCK_N_TILES)
    mask = offs < N_TILES
    partials = tl.load(partial_ptr + c * N_TILES + offs, mask=mask, other=0.0)
    scatter_sum = tl.sum(partials, axis=0)

    # Part A: masked positions contribute full_val each
    mask_count = tl.load(mask_count_ptr + c)
    part_a = full_val * mask_count

    # Total = Part A (masked positions * full_val) + Part B (scatter contributions at unmasked positions)
    # But wait -- positions in the output that receive NO scatter contribution
    # (i.e., positions in [0, 3025) that no source maps to) remain zero.
    # When NOT masked, they contribute 0 to the sum (already accounted for).
    # When masked, they contribute full_val (already in Part A).
    tl.store(out_ptr + c, scatter_sum + part_a)


# ---------------------------------------------------------------------------
# Oracle entry point
# ---------------------------------------------------------------------------

def oracle_fused_scatter_reduce(getitem_9, arg10_1, arg25_1, full,
                                _shape_param_0, _shape_param_1, _shape_param_2):
    """Fused scatter-reduce oracle that avoids the [65536, 3025] intermediate.

    Args:
        getitem_9: f32[1024, 64, 27, 27] - source values
        arg10_1: i8[1024, 64, 27, 27] - max pool offsets
        arg25_1: b8[1024, 64, 55, 55] - where mask
        full: f32[] - scalar fill value
        _shape_param_0, _shape_param_1, _shape_param_2: shape params (unused)

    Returns:
        f32[64] - channel-wise sum
    """
    N, C, src_H, src_W = getitem_9.shape  # [1024, 64, 27, 27]
    out_H, out_W = 55, 55
    device = getitem_9.device
    full_val = full.item()

    total_src_per_channel = N * src_H * src_W  # 1024 * 27 * 27 = 746496

    # Pre-compute Part A: count of True positions in mask per channel
    # arg25_1 shape: [1024, 64, 55, 55], sum over [0, 2, 3] -> [64]
    mask_true_count = arg25_1.sum(dim=(0, 2, 3)).float()  # [64]

    # Tuning parameters
    N_TILES = 64  # 64 channels * 64 tiles = 4096 programs
    BLOCK_SIZE = 2048

    # Round N_TILES up to next power of 2 for finalize kernel
    BLOCK_N_TILES = N_TILES  # already power of 2

    partial = torch.empty(C, N_TILES, dtype=torch.float32, device=device)

    _fused_scatter_reduce_kernel[(C, N_TILES)](
        getitem_9, arg10_1, arg25_1,
        partial,
        N=N, C=C,
        src_H=src_H, src_W=src_W,
        out_H=out_H, out_W=out_W,
        stride_h=2, stride_w=2,
        kernel_w=3,
        total_src_per_channel=total_src_per_channel,
        N_TILES=N_TILES,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    out = torch.empty(C, dtype=torch.float32, device=device)

    _finalize_kernel[(C,)](
        partial, mask_true_count, out,
        full_val,
        N_TILES=N_TILES,
        BLOCK_N_TILES=BLOCK_N_TILES,
    )

    return out


# ---------------------------------------------------------------------------
# Pure PyTorch reference (avoids intermediate, for validation)
# ---------------------------------------------------------------------------

def oracle_pytorch_fused(getitem_9, arg10_1, arg25_1, full,
                         _shape_param_0, _shape_param_1, _shape_param_2):
    """PyTorch gather-mask-reduce implementation (no large intermediate)."""
    N, C, src_H, src_W = getitem_9.shape  # [1024, 64, 27, 27]
    out_H, out_W = 55, 55
    full_val = full.item()

    # Compute max pool indices from offsets
    indices_abs = torch.ops.prims._low_memory_max_pool_offsets_to_indices(
        arg10_1, [3, 3], [55, 55], [2, 2], [0, 0], [1, 1]
    )  # [1024, 64, 27, 27] with values in [0, 3025)

    # Decode to (out_h, out_w)
    idx_h = indices_abs // out_W  # [1024, 64, 27, 27]
    idx_w = indices_abs % out_W   # [1024, 64, 27, 27]

    # Part A: masked positions contribute full_val
    mask_true_count = arg25_1.sum(dim=(0, 2, 3)).float()  # [64]
    part_a = full_val * mask_true_count

    # Part B: gather mask at scatter destinations, accumulate unmasked sources
    # Flatten mask spatially: [1024, 64, 55*55]
    mask_flat = arg25_1.reshape(N, C, out_H * out_W)  # [1024, 64, 3025]

    # Compute flat destination indices
    flat_idx = (idx_h * out_W + idx_w).long()  # [1024, 64, 27, 27]
    flat_idx_reshaped = flat_idx.reshape(N, C, src_H * src_W)  # [1024, 64, 729]

    # Gather mask values at destination positions
    mask_at_dst = torch.gather(mask_flat, 2, flat_idx_reshaped)  # [1024, 64, 729]

    # Source values where destination is NOT masked
    src_flat = getitem_9.reshape(N, C, src_H * src_W)  # [1024, 64, 729]
    unmasked_src = torch.where(mask_at_dst, torch.zeros_like(src_flat), src_flat)
    part_b = unmasked_src.sum(dim=(0, 2))  # [64]

    return part_a + part_b


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
    """Compare oracle output against the reference repro."""
    inputs = make_inputs(device)
    module = _load_repro_module()

    with torch.no_grad():
        # Reference: run the full repro
        ref_out = module.Repro()(*inputs)

        # Oracle Triton kernel
        oracle_out = oracle_fused_scatter_reduce(*inputs)

        # Also verify PyTorch oracle
        pytorch_oracle_out = oracle_pytorch_fused(*inputs)

    max_diff = (oracle_out.float() - ref_out.float()).abs().max().item()
    rel_diff = ((oracle_out.float() - ref_out.float()).abs() /
                (ref_out.float().abs() + 1e-8)).max().item()
    close = torch.allclose(oracle_out.float(), ref_out.float(), rtol=rtol, atol=atol)
    print(f"  Triton oracle: shape={list(oracle_out.shape)}, max_abs_diff={max_diff:.6g}, "
          f"max_rel_diff={rel_diff:.6g}, allclose={close}")

    max_diff_pt = (pytorch_oracle_out.float() - ref_out.float()).abs().max().item()
    close_pt = torch.allclose(pytorch_oracle_out.float(), ref_out.float(), rtol=rtol, atol=atol)
    print(f"  PyTorch oracle: max_abs_diff={max_diff_pt:.6g}, allclose={close_pt}")

    overall = close and close_pt
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_oracle(device: torch.device, warmup: int = 25, rep: int = 100):
    """Benchmark oracle vs torch.compile."""
    inputs = make_inputs(device)
    module = _load_repro_module()

    with torch.no_grad():
        # Warmup oracle
        for _ in range(3):
            oracle_fused_scatter_reduce(*inputs)
        torch.cuda.synchronize()

        # Benchmark Triton oracle
        oracle_ms = triton.testing.do_bench(
            lambda: oracle_fused_scatter_reduce(*inputs),
            warmup=warmup,
            rep=rep,
        )
        oracle_us = oracle_ms * 1000.0

        # Benchmark torch.compile
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
        import torch._inductor.config as inductor_config
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

        # Memory bandwidth analysis
        # Oracle reads: getitem_9 (f32, 1024*64*27*27*4) + arg10_1 (i8, 1024*64*27*27*1)
        #              + arg25_1 (b8, random access per source, upper bound 1024*64*27*27*1)
        # vs Original: reads same inputs + writes/reads [65536, 3025]*4 intermediate
        src_elements = 1024 * 64 * 27 * 27
        bytes_read_oracle = src_elements * 4 + src_elements * 1 + src_elements * 1  # src + offsets + mask lookups
        bytes_read_oracle += 64 * 4  # mask_true_count (negligible)
        bytes_written_oracle = 64 * 4  # output (negligible)
        total_bytes_oracle = bytes_read_oracle + bytes_written_oracle

        intermediate_bytes = 65536 * 3025 * 4  # ~800 MB saved
        print(f"\n{'='*60}")
        print(f"Benchmark results for {REPRO_ID}")
        print(f"{'='*60}")
        print(f"  Oracle (fused scatter-reduce):        {oracle_us:.1f} us")
        print(f"  torch.compile:                        {compile_us:.1f} us")
        print(f"  torch.compile + coord_descent:        {compile_cd_us:.1f} us")
        print(f"  Speedup (oracle vs compile):          {compile_us/oracle_us:.2f}x")
        print(f"  Speedup (oracle vs coord_descent):    {compile_cd_us/oracle_us:.2f}x")
        print(f"  Oracle data volume:                   {total_bytes_oracle/1e6:.1f} MB")
        print(f"  Avoided intermediate:                 {intermediate_bytes/1e6:.1f} MB")
        bw_oracle = total_bytes_oracle / (oracle_ms * 1e-3) / 1e9
        print(f"  Oracle effective bandwidth:           {bw_oracle:.1f} GB/s")
        print(f"{'='*60}")

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
        print(f"\nBenchmark ({REPRO_ID}):")
        benchmark_oracle(device, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
