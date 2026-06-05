"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileViT SiLU-backward plus batch-norm-backward fragment by sharing the fused SiLU derivative producer across both channel reductions, then sinks the finalized per-channel summaries into the returned full f32[128,16,128,128] gradient tensor and f32[16] side vector, whereas Inductor currently schedules the broadcasted sigmoid/Silu-backward producer, sibling sum([0, 2, 3]) reductions, and dependent full-tensor epilogue as separate generic regions over large intermediates; Inductor cannot do this today because its scheduler/codegen lacks a full-scope multi-output reduction template that keeps compatible sibling channel reductions and their dependent materializing epilogue in one fused plan; the fix is SCHEDULER_FUSION: add scheduler/codegen support for shared channel reductions with finalized-scalar epilogues that write both tensor and vector outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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

N = 128
C = 16
H = 128
W = 128
HW = H * W
NHW = N * HW
INV_NHW = 4.76837158203125e-07
BLOCK_ELEMS = 4096
N_TILES = triton.cdiv(NHW, BLOCK_ELEMS) if triton is not None else 512
BLOCK_TILES = 512
OUT0_SHAPE = (N, C, H, W)
OUT0_STRIDE = (C * HW, HW, W, 1)
OUT1_SHAPE = (C,)
OUT1_STRIDE = (1,)
NC11_SHAPE = (1, C, 1, 1)
NC11_STRIDE = (C, 1, 1, 1)
CLASSIFICATION = "SCHEDULER_FUSION"

if triton is not None:

    @triton.jit
    def _partial_silu_bn_sums_kernel(
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        grad_ptr,
        producer_ptr,
        partial_sum_ptr,
        partial_centered_sum_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        NHW_: tl.constexpr,
        BLOCK_ELEMS_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        linear = tile * BLOCK_ELEMS_ + tl.arange(0, BLOCK_ELEMS_)
        mask = linear < NHW_
        n = linear // HW_
        hw = linear - n * HW_
        offsets = n * C_ * HW_ + c * HW_ + hw

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        weight = tl.load(weight_ptr + c).to(tl.float32)
        bias = tl.load(bias_ptr + c).to(tl.float32)

        centered = x - mean
        affine = centered * invstd * weight + bias
        sigmoid = tl.sigmoid(affine)
        producer = grad * sigmoid * (affine * (1.0 - sigmoid) + 1.0)
        active_producer = tl.where(mask, producer, 0.0)

        tl.store(producer_ptr + offsets, producer, mask=mask)
        partial_offset = c * tl.num_programs(1) + tile
        tl.store(partial_sum_ptr + partial_offset, tl.sum(active_producer, axis=0))
        tl.store(
            partial_centered_sum_ptr + partial_offset,
            tl.sum(tl.where(mask, producer * centered, 0.0), axis=0),
        )

    @triton.jit
    def _finalize_silu_bn_sums_kernel(
        partial_sum_ptr,
        partial_centered_sum_ptr,
        invstd_ptr,
        stats_ptr,
        out1_ptr,
        C_: tl.constexpr,
        N_TILES_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
        INV_NHW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile_offsets = tl.arange(0, BLOCK_TILES_)
        mask = tile_offsets < N_TILES_
        partial_base = c * N_TILES_

        sum_parts = tl.load(partial_sum_ptr + partial_base + tile_offsets, mask=mask, other=0.0)
        centered_parts = tl.load(
            partial_centered_sum_ptr + partial_base + tile_offsets,
            mask=mask,
            other=0.0,
        )
        sum_grad = tl.sum(sum_parts, axis=0)
        sum_centered = tl.sum(centered_parts, axis=0)

        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        tl.store(stats_ptr + c, sum_grad * INV_NHW_)
        tl.store(stats_ptr + C_ + c, sum_centered * INV_NHW_ * invstd * invstd)
        tl.store(out1_ptr + c, sum_centered * invstd)

    @triton.jit
    def _write_silu_bn_output_kernel(
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        producer_ptr,
        stats_ptr,
        out0_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        NHW_: tl.constexpr,
        BLOCK_ELEMS_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        linear = tile * BLOCK_ELEMS_ + tl.arange(0, BLOCK_ELEMS_)
        mask = linear < NHW_
        n = linear // HW_
        hw = linear - n * HW_
        offsets = n * C_ * HW_ + c * HW_ + hw

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        producer = tl.load(producer_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        weight = tl.load(weight_ptr + c).to(tl.float32)
        mean_term = tl.load(stats_ptr + c).to(tl.float32)
        var_term = tl.load(stats_ptr + C_ + c).to(tl.float32)

        centered = x - mean
        out = (producer - centered * var_term - mean_term) * invstd * weight
        tl.store(out0_ptr + offsets, out, mask=mask)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
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


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    arg151_1, arg152_1, arg153_1, arg2_1, arg3_1, getitem_135 = inputs
    arg151_1 = _require_f32_tensor("arg151_1", arg151_1, OUT0_SHAPE, OUT0_STRIDE)
    arg152_1 = _require_f32_tensor("arg152_1", arg152_1, NC11_SHAPE, NC11_STRIDE)
    arg153_1 = _require_f32_tensor("arg153_1", arg153_1, NC11_SHAPE, NC11_STRIDE)
    arg2_1 = _require_f32_tensor("arg2_1", arg2_1, OUT1_SHAPE, OUT1_STRIDE)
    arg3_1 = _require_f32_tensor("arg3_1", arg3_1, OUT1_SHAPE, OUT1_STRIDE)
    getitem_135 = _require_f32_tensor("getitem_135", getitem_135, OUT0_SHAPE, OUT0_STRIDE)

    device = arg151_1.device
    if any(t.device != device for t in (arg152_1, arg153_1, arg2_1, arg3_1, getitem_135)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return arg151_1, arg152_1, arg153_1, arg2_1, arg3_1, getitem_135


def oracle_forward(inputs):
    """Run the full captured SiLU-backward plus BN-backward scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_silu_bn_backward_full_scope.py")

    arg151_1, arg152_1, arg153_1, arg2_1, arg3_1, getitem_135 = _validate_inputs(inputs)
    device = arg151_1.device

    partial_sum = torch.empty_strided((C, N_TILES), (N_TILES, 1), device=device, dtype=torch.float32)
    partial_centered_sum = torch.empty_strided(
        (C, N_TILES),
        (N_TILES, 1),
        device=device,
        dtype=torch.float32,
    )
    stats = torch.empty_strided((2, C), (C, 1), device=device, dtype=torch.float32)
    producer = torch.empty_strided(OUT0_SHAPE, OUT0_STRIDE, device=device, dtype=torch.float32)
    out0 = torch.empty_strided(OUT0_SHAPE, OUT0_STRIDE, device=device, dtype=torch.float32)
    out1 = torch.empty_strided(OUT1_SHAPE, OUT1_STRIDE, device=device, dtype=torch.float32)

    _partial_silu_bn_sums_kernel[(C, N_TILES)](
        arg151_1,
        arg152_1,
        arg153_1,
        arg2_1,
        arg3_1,
        getitem_135,
        producer,
        partial_sum,
        partial_centered_sum,
        C_=C,
        HW_=HW,
        NHW_=NHW,
        BLOCK_ELEMS_=BLOCK_ELEMS,
        num_warps=8,
        num_stages=4,
    )
    _finalize_silu_bn_sums_kernel[(C,)](
        partial_sum,
        partial_centered_sum,
        arg153_1,
        stats,
        out1,
        C_=C,
        N_TILES_=N_TILES,
        BLOCK_TILES_=BLOCK_TILES,
        INV_NHW_=INV_NHW,
        num_warps=4,
        num_stages=4,
    )
    _write_silu_bn_output_kernel[(C, N_TILES)](
        arg151_1,
        arg152_1,
        arg153_1,
        arg2_1,
        producer,
        stats,
        out0,
        C_=C,
        HW_=HW,
        NHW_=NHW,
        BLOCK_ELEMS_=BLOCK_ELEMS,
        num_warps=8,
        num_stages=4,
    )
    return out0, out1


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
