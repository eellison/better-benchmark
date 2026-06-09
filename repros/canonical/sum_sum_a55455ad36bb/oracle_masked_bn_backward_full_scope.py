"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Inception-style masked batch-norm-backward fragment by sharing the sibling `sum(where)` and `sum(where * centered)` channel reductions, then reuses the finalized per-channel scalars to emit both the returned f32[128,32,149,149] gradient tensor and f32[32] side vector, whereas Inductor schedules the mask producer, sibling reductions, and dependent full-tensor epilogue as separate generic pieces over materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a full-scope multi-output reduction template that keeps compatible channel reductions and their finalized BN-backward epilogue in one fused plan for large NCHW tensors; the fix is SCHEDULER_FUSION: add scheduler/codegen support for masked channel reductions with dependent tensor and vector epilogues."""
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
    oracle_impl,
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
C = 32
H = 149
W = 149
HW = H * W
INV_NHW = 3.5189856312778704e-07
PARTIAL_BLOCK_HW = 1024
OUT_BLOCK_HW = 1024
OUT0_SHAPE = (N, C, H, W)
OUT0_STRIDE = (C * HW, HW, W, 1)
PARAM_SHAPE = (1, C, 1, 1)
PARAM_STRIDE = (C, 1, 1, 1)
CHANNEL_SHAPE = (C,)
CHANNEL_STRIDE = (1,)
SCALAR_SHAPE = ()
SCALAR_STRIDE = ()


if triton is not None:

    @triton.jit
    def _masked_sum_partial_kernel(
        mask_input_ptr,
        full_ptr,
        source_ptr,
        centered_source_ptr,
        mean_ptr,
        partial_where_ptr,
        partial_centered_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n = tl.program_id(1)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        mean = tl.load(mean_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        sum_where = tl.zeros((), dtype=tl.float32)
        sum_centered = tl.zeros((), dtype=tl.float32)

        for start in range(0, HW_, BLOCK_HW_):
            hw = start + hw_offsets
            mask = hw < HW_
            offsets = n * C_ * HW_ + c * HW_ + hw

            mask_input = tl.load(mask_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            centered_source = tl.load(centered_source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

            where_value = tl.where(mask_input <= 0.0, full_value, source)
            centered = centered_source - mean
            active_where = tl.where(mask, where_value, 0.0)

            sum_where += tl.sum(active_where, axis=0)
            sum_centered += tl.sum(tl.where(mask, where_value * centered, 0.0), axis=0)

        partial_offset = c * 128 + n
        tl.store(partial_where_ptr + partial_offset, sum_where)
        tl.store(partial_centered_ptr + partial_offset, sum_centered)

    @triton.jit
    def _masked_sum_write_output_kernel(
        mask_input_ptr,
        full_ptr,
        source_ptr,
        centered_source_ptr,
        mean_ptr,
        invstd_ptr,
        affine_weight_ptr,
        partial_where_ptr,
        partial_centered_ptr,
        out0_ptr,
        out1_ptr,
        C_: tl.constexpr,
        N_: tl.constexpr,
        HW_: tl.constexpr,
        INV_NHW_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n = tl.program_id(1)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        n_offsets = tl.arange(0, BLOCK_N_)
        partial_offsets = c * N_ + n_offsets

        partial_where = tl.load(partial_where_ptr + partial_offsets).to(tl.float32)
        partial_centered = tl.load(partial_centered_ptr + partial_offsets).to(tl.float32)
        sum_where = tl.sum(partial_where, axis=0)
        sum_centered = tl.sum(partial_centered, axis=0)

        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        affine_weight = tl.load(affine_weight_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        mean_term = sum_where * INV_NHW_
        centered_term = sum_centered * INV_NHW_ * invstd * invstd
        output_scale = invstd * affine_weight

        for start in range(0, HW_, BLOCK_HW_):
            hw = start + hw_offsets
            mask = hw < HW_
            offsets = n * C_ * HW_ + c * HW_ + hw

            mask_input = tl.load(mask_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            centered_source = tl.load(centered_source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

            where_value = tl.where(mask_input <= 0.0, full_value, source)
            centered = centered_source - mean
            output = (where_value - centered * centered_term - mean_term) * output_scale
            tl.store(out0_ptr + offsets, output, mask=mask)

        if n == 0:
            tl.store(out1_ptr + c, sum_centered * invstd)


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
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    arg238_1, full_1, getitem_276, arg236_1, arg591_1, arg237_1, arg2_1 = inputs
    arg238_1 = _require_f32_tensor("arg238_1", arg238_1, OUT0_SHAPE, OUT0_STRIDE)
    full_1 = _require_f32_tensor("full_1", full_1, SCALAR_SHAPE, SCALAR_STRIDE)
    getitem_276 = _require_f32_tensor("getitem_276", getitem_276, OUT0_SHAPE, OUT0_STRIDE)
    arg236_1 = _require_f32_tensor("arg236_1", arg236_1, OUT0_SHAPE, OUT0_STRIDE)
    arg591_1 = _require_f32_tensor("arg591_1", arg591_1, PARAM_SHAPE, PARAM_STRIDE)
    arg237_1 = _require_f32_tensor("arg237_1", arg237_1, CHANNEL_SHAPE, CHANNEL_STRIDE)
    arg2_1 = _require_f32_tensor("arg2_1", arg2_1, CHANNEL_SHAPE, CHANNEL_STRIDE)

    device = arg238_1.device
    if any(t.device != device for t in (full_1, getitem_276, arg236_1, arg591_1, arg237_1, arg2_1)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return arg238_1, full_1, getitem_276, arg236_1, arg591_1, arg237_1, arg2_1


@oracle_impl(hardware="H100", shapes="(T([128, 32, 149, 149], f32), T([], f32), T([128, 32, 149, 149], f32), T([128, 32, 149, 149], f32), T([1, 32, 1, 1], f32), T([32], f32), T([32], f32))")
def oracle_forward(inputs):
    """Run the exact Repro.forward scope with shared masked channel reductions."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    arg238_1, full_1, getitem_276, arg236_1, arg591_1, arg237_1, arg2_1 = _validate_inputs(inputs)
    partial_where = torch.empty_strided((C, N), (N, 1), device=arg238_1.device, dtype=torch.float32)
    partial_centered = torch.empty_strided((C, N), (N, 1), device=arg238_1.device, dtype=torch.float32)
    out0 = torch.empty_strided(OUT0_SHAPE, OUT0_STRIDE, device=arg238_1.device, dtype=torch.float32)
    out1 = torch.empty_strided(CHANNEL_SHAPE, CHANNEL_STRIDE, device=arg238_1.device, dtype=torch.float32)

    _masked_sum_partial_kernel[(C, N)](
        arg238_1,
        full_1,
        getitem_276,
        arg236_1,
        arg591_1,
        partial_where,
        partial_centered,
        C_=C,
        HW_=HW,
        BLOCK_HW_=PARTIAL_BLOCK_HW,
        num_warps=8,
        num_stages=4,
    )
    _masked_sum_write_output_kernel[(C, N)](
        arg238_1,
        full_1,
        getitem_276,
        arg236_1,
        arg591_1,
        arg237_1,
        arg2_1,
        partial_where,
        partial_centered,
        out0,
        out1,
        C_=C,
        N_=N,
        HW_=HW,
        INV_NHW_=INV_NHW,
        BLOCK_N_=N,
        BLOCK_HW_=OUT_BLOCK_HW,
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
