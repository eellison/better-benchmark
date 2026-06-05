"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete chained RepVGG batch-norm-backward scope by sharing the first channel reductions, materializing the masked residual producer once, sharing the downstream sibling channel reductions, and emitting both full tensor gradients plus all three vector outputs, whereas Inductor currently schedules the first BN-backward epilogue, mask/add producer, duplicate downstream sums, and two full-output BN-backward epilogues as separate generic regions over large intermediates; Inductor cannot do this today because its scheduler/codegen does not build a full-scope multi-output reduction plan with reusable channel summaries feeding multiple dependent tensor epilogues; the fix is SCHEDULER_FUSION: add scheduler support for chained sibling channel reductions that share finalized summaries and sink them into fused full-tensor/vector epilogues."""
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


N = 128
C = 96
H = 56
W = 56
HW = H * W
NHW = N * HW
SHAPE_4D = (N, C, H, W)
STRIDE_4D = (C * HW, HW, W, 1)
SHAPE_MEAN = (1, C, 1, 1)
STRIDE_MEAN = (C, 1, 1, 1)
SHAPE_VEC = (C,)
STRIDE_VEC = (1,)
INV_NHW = 2.4912308673469386e-06
REDUCE_BLOCK = 4096
FINAL_BLOCK = 128
EPILOGUE_BLOCK = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _first_bn_partial_kernel(
        where_19_ptr,
        arg118_ptr,
        mean1_ptr,
        partial_sum_ptr,
        partial_dot_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        NHW_: tl.constexpr,
        NUM_TILES_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK + tl.arange(0, BLOCK)
        active = k < NHW_
        n = k // HW_
        hw = k - n * HW_
        offsets = n * C_ * HW_ + c * HW_ + hw

        where_value = tl.load(where_19_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        centered = (
            tl.load(arg118_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            - tl.load(mean1_ptr + c).to(tl.float32)
        )
        where_value = tl.where(active, where_value, 0.0)
        centered = tl.where(active, centered, 0.0)
        partial_offset = c * NUM_TILES_ + tile
        tl.store(partial_sum_ptr + partial_offset, tl.sum(where_value, axis=0))
        tl.store(partial_dot_ptr + partial_offset, tl.sum(where_value * centered, axis=0))

    @triton.jit
    def _first_bn_finalize_kernel(
        partial_sum_ptr,
        partial_dot_ptr,
        invstd1_ptr,
        affine1_ptr,
        first_stats_ptr,
        out0_ptr,
        C_: tl.constexpr,
        NUM_TILES_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        INV_NHW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES)
        mask = tiles < NUM_TILES_
        offsets = c * NUM_TILES_ + tiles
        sum_where = tl.sum(tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        sum_dot = tl.sum(tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)

        invstd = tl.load(invstd1_ptr + c).to(tl.float32)
        affine = tl.load(affine1_ptr + c).to(tl.float32)
        tl.store(first_stats_ptr + c, sum_where * INV_NHW_)
        tl.store(first_stats_ptr + C_ + c, sum_dot * INV_NHW_ * invstd * invstd)
        tl.store(first_stats_ptr + 2 * C_ + c, invstd * affine)
        tl.store(out0_ptr + c, sum_dot * invstd)

    @triton.jit
    def _downstream_partial_kernel(
        getitem_114_ptr,
        getitem_117_ptr,
        where_19_ptr,
        arg118_ptr,
        mean1_ptr,
        first_stats_ptr,
        full_ptr,
        arg116_ptr,
        mean2_ptr,
        arg114_ptr,
        mean3_ptr,
        where_self_ptr,
        partial_sum_ptr,
        partial_dot2_ptr,
        partial_dot3_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        NHW_: tl.constexpr,
        NUM_TILES_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK + tl.arange(0, BLOCK)
        active = k < NHW_
        n = k // HW_
        hw = k - n * HW_
        offsets = n * C_ * HW_ + c * HW_ + hw

        add_value = (
            tl.load(getitem_114_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            + tl.load(getitem_117_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        )
        where_value = tl.load(where_19_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        arg118 = tl.load(arg118_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        centered1 = arg118 - tl.load(mean1_ptr + c).to(tl.float32)
        mean_term1 = tl.load(first_stats_ptr + c).to(tl.float32)
        coeff1 = tl.load(first_stats_ptr + C_ + c).to(tl.float32)
        scale1 = tl.load(first_stats_ptr + 2 * C_ + c).to(tl.float32)
        first_grad = (where_value - centered1 * coeff1 - mean_term1) * scale1
        masked_value = tl.where(arg118 <= 0.0, tl.load(full_ptr).to(tl.float32), add_value + first_grad)
        masked_value = tl.where(active, masked_value, 0.0)

        centered2 = (
            tl.load(arg116_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            - tl.load(mean2_ptr + c).to(tl.float32)
        )
        centered3 = (
            tl.load(arg114_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            - tl.load(mean3_ptr + c).to(tl.float32)
        )
        centered2 = tl.where(active, centered2, 0.0)
        centered3 = tl.where(active, centered3, 0.0)

        tl.store(where_self_ptr + offsets, masked_value, mask=active)
        partial_offset = c * NUM_TILES_ + tile
        tl.store(partial_sum_ptr + partial_offset, tl.sum(masked_value, axis=0))
        tl.store(partial_dot2_ptr + partial_offset, tl.sum(masked_value * centered2, axis=0))
        tl.store(partial_dot3_ptr + partial_offset, tl.sum(masked_value * centered3, axis=0))

    @triton.jit
    def _downstream_finalize_kernel(
        partial_sum_ptr,
        partial_dot2_ptr,
        partial_dot3_ptr,
        invstd2_ptr,
        affine2_ptr,
        invstd3_ptr,
        affine3_ptr,
        downstream_stats_ptr,
        out2_ptr,
        out4_ptr,
        C_: tl.constexpr,
        NUM_TILES_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        INV_NHW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES)
        mask = tiles < NUM_TILES_
        offsets = c * NUM_TILES_ + tiles
        sum_where = tl.sum(tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        sum_dot2 = tl.sum(tl.load(partial_dot2_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        sum_dot3 = tl.sum(tl.load(partial_dot3_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)

        invstd2 = tl.load(invstd2_ptr + c).to(tl.float32)
        affine2 = tl.load(affine2_ptr + c).to(tl.float32)
        invstd3 = tl.load(invstd3_ptr + c).to(tl.float32)
        affine3 = tl.load(affine3_ptr + c).to(tl.float32)
        tl.store(downstream_stats_ptr + c, sum_where * INV_NHW_)
        tl.store(downstream_stats_ptr + C_ + c, sum_dot2 * INV_NHW_ * invstd2 * invstd2)
        tl.store(downstream_stats_ptr + 2 * C_ + c, invstd2 * affine2)
        tl.store(downstream_stats_ptr + 3 * C_ + c, sum_dot3 * INV_NHW_ * invstd3 * invstd3)
        tl.store(downstream_stats_ptr + 4 * C_ + c, invstd3 * affine3)
        tl.store(out2_ptr + c, sum_dot2 * invstd2)
        tl.store(out4_ptr + c, sum_dot3 * invstd3)

    @triton.jit
    def _dual_tensor_epilogue_kernel(
        where_self_ptr,
        arg116_ptr,
        mean2_ptr,
        arg114_ptr,
        mean3_ptr,
        downstream_stats_ptr,
        out1_ptr,
        out3_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        NHW_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK + tl.arange(0, BLOCK)
        active = k < NHW_
        n = k // HW_
        hw = k - n * HW_
        offsets = n * C_ * HW_ + c * HW_ + hw

        where_value = tl.load(where_self_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        mean_term = tl.load(downstream_stats_ptr + c).to(tl.float32)

        centered2 = (
            tl.load(arg116_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            - tl.load(mean2_ptr + c).to(tl.float32)
        )
        coeff2 = tl.load(downstream_stats_ptr + C_ + c).to(tl.float32)
        scale2 = tl.load(downstream_stats_ptr + 2 * C_ + c).to(tl.float32)
        out2_value = (where_value - centered2 * coeff2 - mean_term) * scale2

        centered3 = (
            tl.load(arg114_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            - tl.load(mean3_ptr + c).to(tl.float32)
        )
        coeff3 = tl.load(downstream_stats_ptr + 3 * C_ + c).to(tl.float32)
        scale3 = tl.load(downstream_stats_ptr + 4 * C_ + c).to(tl.float32)
        out3_value = (where_value - centered3 * coeff3 - mean_term) * scale3

        tl.store(out1_ptr + offsets, out2_value, mask=active)
        tl.store(out3_ptr + offsets, out3_value, mask=active)


def _require_f32_tensor(name, value, shape, stride):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_full_scope_bn_chain.py")
    if len(inputs) != 16:
        raise ValueError(f"{REPRO_ID} expects 16 inputs, got {len(inputs)}")

    (
        getitem_114,
        getitem_117,
        where_19,
        arg118_1,
        arg292_1,
        arg119_1,
        arg9_1,
        full_1,
        arg116_1,
        arg293_1,
        arg117_1,
        arg8_1,
        arg114_1,
        arg294_1,
        arg115_1,
        arg6_1,
    ) = inputs

    tensors_4d = (
        ("getitem_114", getitem_114),
        ("getitem_117", getitem_117),
        ("where_19", where_19),
        ("arg118_1", arg118_1),
        ("arg116_1", arg116_1),
        ("arg114_1", arg114_1),
    )
    tensors_mean = (
        ("arg292_1", arg292_1),
        ("arg293_1", arg293_1),
        ("arg294_1", arg294_1),
    )
    tensors_vec = (
        ("arg119_1", arg119_1),
        ("arg9_1", arg9_1),
        ("arg117_1", arg117_1),
        ("arg8_1", arg8_1),
        ("arg115_1", arg115_1),
        ("arg6_1", arg6_1),
    )
    for name, tensor in tensors_4d:
        _require_f32_tensor(name, tensor, SHAPE_4D, STRIDE_4D)
    for name, tensor in tensors_mean:
        _require_f32_tensor(name, tensor, SHAPE_MEAN, STRIDE_MEAN)
    for name, tensor in tensors_vec:
        _require_f32_tensor(name, tensor, SHAPE_VEC, STRIDE_VEC)
    _require_f32_tensor("full_1", full_1, (), ())

    device = getitem_114.device
    checked_tensors = (
        getitem_117,
        where_19,
        arg118_1,
        arg292_1,
        arg119_1,
        arg9_1,
        full_1,
        arg116_1,
        arg293_1,
        arg117_1,
        arg8_1,
        arg114_1,
        arg294_1,
        arg115_1,
        arg6_1,
    )
    if any(t.device != device for t in checked_tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return (
        getitem_114,
        getitem_117,
        where_19,
        arg118_1,
        arg292_1,
        arg119_1,
        arg9_1,
        full_1,
        arg116_1,
        arg293_1,
        arg117_1,
        arg8_1,
        arg114_1,
        arg294_1,
        arg115_1,
        arg6_1,
    )


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
    (
        getitem_114,
        getitem_117,
        where_19,
        arg118_1,
        arg292_1,
        arg119_1,
        arg9_1,
        full_1,
        arg116_1,
        arg293_1,
        arg117_1,
        arg8_1,
        arg114_1,
        arg294_1,
        arg115_1,
        arg6_1,
    ) = _validate_inputs(tuple(inputs))

    device = getitem_114.device
    num_tiles = triton.cdiv(NHW, REDUCE_BLOCK)
    block_tiles = triton.next_power_of_2(num_tiles)

    partial_first_sum = torch.empty_strided((C, num_tiles), (num_tiles, 1), device=device, dtype=torch.float32)
    partial_first_dot = torch.empty_strided((C, num_tiles), (num_tiles, 1), device=device, dtype=torch.float32)
    first_stats = torch.empty_strided((3, C), (C, 1), device=device, dtype=torch.float32)
    out0 = torch.empty_strided(SHAPE_VEC, STRIDE_VEC, device=device, dtype=torch.float32)

    _first_bn_partial_kernel[(C, num_tiles)](
        where_19,
        arg118_1,
        arg292_1,
        partial_first_sum,
        partial_first_dot,
        C_=C,
        HW_=HW,
        NHW_=NHW,
        NUM_TILES_=num_tiles,
        BLOCK=REDUCE_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _first_bn_finalize_kernel[(C,)](
        partial_first_sum,
        partial_first_dot,
        arg119_1,
        arg9_1,
        first_stats,
        out0,
        C_=C,
        NUM_TILES_=num_tiles,
        BLOCK_TILES=block_tiles,
        INV_NHW_=INV_NHW,
        num_warps=8,
        num_stages=4,
    )

    where_self = torch.empty_strided(SHAPE_4D, STRIDE_4D, device=device, dtype=torch.float32)
    partial_sum = torch.empty_strided((C, num_tiles), (num_tiles, 1), device=device, dtype=torch.float32)
    partial_dot2 = torch.empty_strided((C, num_tiles), (num_tiles, 1), device=device, dtype=torch.float32)
    partial_dot3 = torch.empty_strided((C, num_tiles), (num_tiles, 1), device=device, dtype=torch.float32)

    _downstream_partial_kernel[(C, num_tiles)](
        getitem_114,
        getitem_117,
        where_19,
        arg118_1,
        arg292_1,
        first_stats,
        full_1,
        arg116_1,
        arg293_1,
        arg114_1,
        arg294_1,
        where_self,
        partial_sum,
        partial_dot2,
        partial_dot3,
        C_=C,
        HW_=HW,
        NHW_=NHW,
        NUM_TILES_=num_tiles,
        BLOCK=REDUCE_BLOCK,
        num_warps=8,
        num_stages=4,
    )

    downstream_stats = torch.empty_strided((5, C), (C, 1), device=device, dtype=torch.float32)
    out2 = torch.empty_strided(SHAPE_VEC, STRIDE_VEC, device=device, dtype=torch.float32)
    out4 = torch.empty_strided(SHAPE_VEC, STRIDE_VEC, device=device, dtype=torch.float32)
    _downstream_finalize_kernel[(C,)](
        partial_sum,
        partial_dot2,
        partial_dot3,
        arg117_1,
        arg8_1,
        arg115_1,
        arg6_1,
        downstream_stats,
        out2,
        out4,
        C_=C,
        NUM_TILES_=num_tiles,
        BLOCK_TILES=block_tiles,
        INV_NHW_=INV_NHW,
        num_warps=8,
        num_stages=4,
    )

    out1 = torch.empty_strided(SHAPE_4D, STRIDE_4D, device=device, dtype=torch.float32)
    out3 = torch.empty_strided(SHAPE_4D, STRIDE_4D, device=device, dtype=torch.float32)
    _dual_tensor_epilogue_kernel[(C, triton.cdiv(NHW, EPILOGUE_BLOCK))](
        where_self,
        arg116_1,
        arg293_1,
        arg114_1,
        arg294_1,
        downstream_stats,
        out1,
        out3,
        C_=C,
        HW_=HW,
        NHW_=NHW,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    return out0, out1, out2, out3, out4


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
