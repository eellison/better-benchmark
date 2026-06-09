"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full RegNet BN-backward return by fusing the masked squeeze-excitation gradient producer into two sibling per-channel reductions split across N/H/W, then reusing the finalized sums for the dependent BN epilogue tensor and side vector, whereas Inductor materializes/schedules the producer, the two sum([0, 2, 3]) reductions, and the reduction-dependent pointwise output as separate generic kernels; Inductor cannot do this today because scheduler/codegen lacks a cooperative split-K multi-output channel-reduction template that can share a fused producer across reductions and feed a full-tensor epilogue; the fix is COOPERATIVE_SPLIT_K: add split-K multi-output channel reductions with fused producer and dependent epilogue support."""
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

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


N = 32
C = 224
H = 56
W = 56
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
REDUCE_SCALE = 9.964923469387754e-06


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_dual_sum_kernel(
        getitem_ptr,
        sigmoid_ptr,
        bias_ptr,
        relu_ptr,
        full_ptr,
        centered_ptr,
        mean_ptr,
        partial_x_ptr,
        partial_x_centered_ptr,
        num_tiles: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        TOTAL_SPATIAL_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = k < TOTAL_SPATIAL_

        n = k // HW_
        hw = k - n * HW_
        spatial_offsets = n * (C_ * HW_) + c * HW_ + hw
        channel_offsets = n * C_ + c

        getitem = tl.load(getitem_ptr + spatial_offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid = tl.load(sigmoid_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        relu = tl.load(relu_ptr + spatial_offsets, mask=mask, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        centered = tl.load(centered_ptr + spatial_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c).to(tl.float32)

        x = getitem * sigmoid + bias * 0.00031887755102040814
        x = tl.where(relu <= 0.0, full_value, x)
        centered = centered - mean

        partial_offset = c * num_tiles + tile
        tl.store(partial_x_ptr + partial_offset, tl.sum(x, axis=0))
        tl.store(partial_x_centered_ptr + partial_offset, tl.sum(x * centered, axis=0))


    @triton.jit
    def _finalize_dual_sum_kernel(
        partial_x_ptr,
        partial_x_centered_ptr,
        invstd_ptr,
        sum_x_ptr,
        sum_x_centered_ptr,
        vector_out_ptr,
        num_tiles: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        c = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES)
        mask = tiles < num_tiles
        offsets = c * num_tiles + tiles

        x_values = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        xc_values = tl.load(partial_x_centered_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.sum(x_values, axis=0)
        sum_x_centered = tl.sum(xc_values, axis=0)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)

        tl.store(sum_x_ptr + c, sum_x)
        tl.store(sum_x_centered_ptr + c, sum_x_centered)
        tl.store(vector_out_ptr + c, sum_x_centered * invstd)


    @triton.jit
    def _bn_epilogue_kernel(
        getitem_ptr,
        sigmoid_ptr,
        bias_ptr,
        relu_ptr,
        full_ptr,
        centered_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        sum_x_ptr,
        sum_x_centered_ptr,
        out_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        NUMEL_: tl.constexpr,
        REDUCE_SCALE_: tl.constexpr,
        BLOCK_ELEMS: tl.constexpr,
    ):
        linear = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
        mask = linear < NUMEL_

        hw = linear % HW_
        c = (linear // HW_) % C_
        n = linear // (C_ * HW_)
        spatial_offsets = n * (C_ * HW_) + c * HW_ + hw
        channel_offsets = n * C_ + c

        getitem = tl.load(getitem_ptr + spatial_offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid = tl.load(sigmoid_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        relu = tl.load(relu_ptr + spatial_offsets, mask=mask, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        centered = tl.load(centered_ptr + spatial_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.load(sum_x_ptr + c, mask=mask, other=0.0).to(tl.float32)
        sum_x_centered = tl.load(sum_x_centered_ptr + c, mask=mask, other=0.0).to(tl.float32)

        x = getitem * sigmoid + bias * 0.00031887755102040814
        x = tl.where(relu <= 0.0, full_value, x)
        centered = centered - mean
        mean_term = sum_x * REDUCE_SCALE_
        variance_term = sum_x_centered * REDUCE_SCALE_ * invstd * invstd
        out = (x - centered * variance_term - mean_term) * (invstd * weight)
        tl.store(out_ptr + linear, out, mask=mask)


def _check_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")


@oracle_impl(hardware="H100", shapes="(T([32, 224, 56, 56], f32), T([32, 224, 1, 1], f32), T([32, 224, 1, 1], f32), T([32, 224, 56, 56], f32), T([], f32), T([1, 224, 1, 1], f32), T([32, 224, 56, 56], f32), T([1, 224, 1, 1], f32), T([224], f32), S([32, 224, 56, 56]))")
def oracle_forward(inputs):
    """Run the full oracle computation for Repro.forward."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        getitem_282,
        sigmoid_18,
        getitem_288,
        relu_19,
        full,
        arg192_1,
        arg191_1,
        arg193_1,
        arg6_1,
        shape_param_0,
    ) = inputs

    if tuple(shape_param_0) != (N, C, H, W):
        raise ValueError(f"unexpected shape parameter for {REPRO_ID}: {shape_param_0}")
    _check_tensor("getitem_282", getitem_282, (N, C, H, W))
    _check_tensor("sigmoid_18", sigmoid_18, (N, C, 1, 1))
    _check_tensor("getitem_288", getitem_288, (N, C, 1, 1))
    _check_tensor("relu_19", relu_19, (N, C, H, W))
    _check_tensor("full", full, ())
    _check_tensor("arg192_1", arg192_1, (1, C, 1, 1))
    _check_tensor("arg191_1", arg191_1, (N, C, H, W))
    _check_tensor("arg193_1", arg193_1, (1, C, 1, 1))
    _check_tensor("arg6_1", arg6_1, (C,))

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_x = torch.empty((C, num_tiles), device=getitem_282.device, dtype=torch.float32)
    partial_x_centered = torch.empty((C, num_tiles), device=getitem_282.device, dtype=torch.float32)
    sum_x = torch.empty((C,), device=getitem_282.device, dtype=torch.float32)
    sum_x_centered = torch.empty((C,), device=getitem_282.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=getitem_282.device, dtype=getitem_282.dtype)

    _partial_dual_sum_kernel[(C, num_tiles)](
        getitem_282,
        sigmoid_18,
        getitem_288,
        relu_19,
        full,
        arg191_1,
        arg192_1,
        partial_x,
        partial_x_centered,
        num_tiles=num_tiles,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_K=block_k,
        num_warps=4,
    )

    block_tiles = triton.next_power_of_2(num_tiles)
    _finalize_dual_sum_kernel[(C,)](
        partial_x,
        partial_x_centered,
        arg193_1,
        sum_x,
        sum_x_centered,
        vector_out,
        num_tiles=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=4,
    )

    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=getitem_282.device,
        dtype=getitem_282.dtype,
    )
    block_elems = 256
    _bn_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        getitem_282,
        sigmoid_18,
        getitem_288,
        relu_19,
        full,
        arg191_1,
        arg192_1,
        arg193_1,
        arg6_1,
        sum_x,
        sum_x_centered,
        out,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        REDUCE_SCALE_=REDUCE_SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out, vector_out


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

    inputs = get_inputs()
    instance = get_repro_instance()

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
