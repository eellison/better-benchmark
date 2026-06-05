"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 GRN-style backward tail returned by Repro.forward, including both NCHW-to-NHWC permutes, the per-position channel reductions, the dependent input-gradient epilogue, and all three channel-gradient reductions in one split reduction/finalize plan, whereas Inductor currently schedules the permute views, inner sums, sibling outer sums, and dependent epilogue as generic reduction/pointwise work with avoidable rereads and accumulator overhead; Inductor cannot do this today because its scheduler/codegen does not form a full-scope multi-output reduction template that shares row-local channel summaries with same-axis channel accumulators; the fix is SCHEDULER_FUSION: add a guarded multi-output reduction schedule that keeps the layout views virtual, computes row scalars once, and fuses compatible channel reductions plus epilogues."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _partial_reduce_kernel(
        getitem_ptr,
        weight_ptr,
        arg85_ptr,
        mean_ptr,
        invstd_ptr,
        partial0_ptr,
        partial1_ptr,
        partial2_ptr,
        TOTAL_POS: tl.constexpr,
        C: tl.constexpr,
        HW: tl.constexpr,
        BLOCK_POS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        tile = tl.program_id(0)
        pos = tile * BLOCK_POS + tl.arange(0, BLOCK_POS)
        channels = tl.arange(0, BLOCK_C)

        n = pos // HW
        hw = pos - n * HW
        nchw_offsets = n[:, None] * (C * HW) + channels[None, :] * HW + hw[:, None]
        mask = (pos[:, None] < TOTAL_POS) & (channels[None, :] < C)

        x = tl.load(getitem_ptr + nchw_offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(arg85_ptr + nchw_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channels, mask=channels < C, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + pos, mask=pos < TOTAL_POS, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + pos, mask=pos < TOTAL_POS, other=0.0).to(tl.float32)

        weighted_x = x * weight[None, :]
        normalized = (y - mean[:, None]) * invstd[:, None]
        row_sum = tl.sum(weighted_x, axis=1)
        row_dot = tl.sum(weighted_x * normalized, axis=1)

        grad = (invstd[:, None] * 0.0125) * (
            weighted_x * 80.0 - row_sum[:, None] - normalized * row_dot[:, None]
        )

        out0 = tl.sum(x * normalized, axis=0)
        out1 = tl.sum(x, axis=0)
        out2 = tl.sum(grad, axis=0)

        out_offsets = tile * C + channels
        tl.store(partial0_ptr + out_offsets, out0, mask=channels < C)
        tl.store(partial1_ptr + out_offsets, out1, mask=channels < C)
        tl.store(partial2_ptr + out_offsets, out2, mask=channels < C)

    @triton.jit
    def _finalize_reduce_kernel(
        partial0_ptr,
        partial1_ptr,
        partial2_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        N_TILES: tl.constexpr,
        C: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_TILES)

        acc0 = tl.zeros((BLOCK_TILES,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_TILES,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_TILES,), dtype=tl.float32)

        for start in range(0, N_TILES, BLOCK_TILES):
            tile_offsets = start + offsets
            mask = tile_offsets < N_TILES
            idx = tile_offsets * C + channel
            acc0 += tl.load(partial0_ptr + idx, mask=mask, other=0.0)
            acc1 += tl.load(partial1_ptr + idx, mask=mask, other=0.0)
            acc2 += tl.load(partial2_ptr + idx, mask=mask, other=0.0)

        tl.store(out0_ptr + channel, tl.sum(acc0, axis=0))
        tl.store(out1_ptr + channel, tl.sum(acc1, axis=0))
        tl.store(out2_ptr + channel, tl.sum(acc2, axis=0))


def _reference_forward(inputs):
    getitem_129, arg4_1, arg85_1, arg86_1, arg87_1 = inputs
    permute_default = torch.ops.aten.permute.default(getitem_129, [0, 2, 3, 1])
    mul_tensor = torch.ops.aten.mul.Tensor(permute_default, arg4_1)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 80)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
    permute_default_1 = torch.ops.aten.permute.default(arg85_1, [0, 2, 3, 1])
    sub_tensor = torch.ops.aten.sub.Tensor(permute_default_1, arg86_1)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, arg87_1)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2)
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [3], True)
    mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1)
    sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list)
    sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4)
    div_tensor = torch.ops.aten.div.Tensor(arg87_1, 80)
    mul_tensor_5 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2)
    mul_tensor_6 = torch.ops.aten.mul.Tensor(permute_default, mul_tensor_2)
    sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1, 2])
    sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(permute_default, [0, 1, 2])
    permute_default_2 = torch.ops.aten.permute.default(mul_tensor_5, [0, 3, 1, 2])
    sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(permute_default_2, [0, 2, 3])
    return (sum_dim_int_list_2, sum_dim_int_list_3, sum_dim_int_list_4)


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    getitem_129, arg4_1, arg85_1, arg86_1, arg87_1 = inputs
    if triton is None or not getitem_129.is_cuda:
        return _reference_forward(inputs)

    n, channels, height, width = getitem_129.shape
    total_pos = n * height * width
    hw = height * width
    block_pos = 64
    block_channels = 1 << (channels - 1).bit_length()
    n_tiles = triton.cdiv(total_pos, block_pos)

    partial0 = torch.empty((n_tiles, channels), device=getitem_129.device, dtype=torch.float32)
    partial1 = torch.empty((n_tiles, channels), device=getitem_129.device, dtype=torch.float32)
    partial2 = torch.empty((n_tiles, channels), device=getitem_129.device, dtype=torch.float32)
    out0 = torch.empty((channels,), device=getitem_129.device, dtype=torch.float32)
    out1 = torch.empty((channels,), device=getitem_129.device, dtype=torch.float32)
    out2 = torch.empty((channels,), device=getitem_129.device, dtype=torch.float32)

    _partial_reduce_kernel[(n_tiles,)](
        getitem_129,
        arg4_1,
        arg85_1,
        arg86_1,
        arg87_1,
        partial0,
        partial1,
        partial2,
        TOTAL_POS=total_pos,
        C=channels,
        HW=hw,
        BLOCK_POS=block_pos,
        BLOCK_C=block_channels,
        num_warps=4,
    )
    _finalize_reduce_kernel[(channels,)](
        partial0,
        partial1,
        partial2,
        out0,
        out1,
        out2,
        N_TILES=n_tiles,
        C=channels,
        BLOCK_TILES=512,
        num_warps=8,
    )
    return (out0, out1, out2)


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
