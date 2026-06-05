"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full EfficientNet batch-norm-backward fragment by sharing the fused gradient producer across both channel reductions and using the finalized per-channel sums to emit the returned `[128, 32, 112, 112]` gradient tensor and `[32]` side gradient vector, whereas Inductor currently schedules the sigmoid/broadcast/reciprocal pointwise producer, sibling `sum([0, 2, 3])` reductions, and dependent BN-backward epilogue as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a full-scope multi-output reduction template that keeps compatible sibling channel reductions and their dependent full-tensor epilogue in one fused plan; the fix is SCHEDULER_FUSION: add scheduler/codegen support for shared channel reductions with a finalized-scalar epilogue that writes both the tensor and vector outputs."""
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
C = 32
H = 112
W = 112
HW = H * W
INV_HW = 1.0 / HW
INV_NHW = 1.0 / (N * HW)
BLOCK_HW = 2048
OUT0_SHAPE = (N, C, H, W)
OUT0_STRIDE = (C * HW, HW, W, 1)
OUT1_SHAPE = (C,)
OUT1_STRIDE = (1,)
NC11_SHAPE = (N, C, 1, 1)
NC11_STRIDE = (C, 1, 1, 1)
MEAN_SHAPE = (1, C, 1, 1)
MEAN_STRIDE = (C, 1, 1, 1)


if triton is not None:

    @triton.jit
    def _partial_bn_sums_kernel(
        getitem_228_ptr,
        sigmoid_15_ptr,
        getitem_234_ptr,
        add_148_ptr,
        add_147_ptr,
        arg170_ptr,
        arg169_ptr,
        producer_ptr,
        partial_sum_ptr,
        partial_centered_sum_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n = tl.program_id(1)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        nc_offset = n * C_ + c
        sigmoid = tl.load(sigmoid_15_ptr + nc_offset).to(tl.float32)
        pooled = tl.load(getitem_234_ptr + nc_offset).to(tl.float32)
        mean = tl.load(arg170_ptr + c).to(tl.float32)

        sum_grad = tl.zeros((), dtype=tl.float32)
        sum_centered = tl.zeros((), dtype=tl.float32)

        for start in range(0, HW_, BLOCK_HW_):
            hw = start + hw_offsets
            mask = hw < HW_
            offsets = n * C_ * HW_ + c * HW_ + hw

            grad = tl.load(getitem_228_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            inv_denom = 1.0 / tl.load(add_148_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
            gate = tl.load(add_147_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            activation = tl.load(arg169_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

            producer = (grad * sigmoid + pooled * INV_HW_) * inv_denom
            producer = producer * (gate * (1.0 - inv_denom) + 1.0)
            centered = activation - mean

            active_producer = tl.where(mask, producer, 0.0)
            sum_grad += tl.sum(active_producer, axis=0)
            sum_centered += tl.sum(tl.where(mask, producer * centered, 0.0), axis=0)
            tl.store(producer_ptr + offsets, producer, mask=mask)

        partial_offset = c * 128 + n
        tl.store(partial_sum_ptr + partial_offset, sum_grad)
        tl.store(partial_centered_sum_ptr + partial_offset, sum_centered)

    @triton.jit
    def _finalize_bn_sums_kernel(
        partial_sum_ptr,
        partial_centered_sum_ptr,
        arg171_ptr,
        stats_ptr,
        out1_ptr,
        C_: tl.constexpr,
        N_: tl.constexpr,
        INV_NHW_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n_offsets = tl.arange(0, BLOCK_N_)
        partial_offsets = c * N_ + n_offsets
        mask = n_offsets < N_

        grad_parts = tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
        centered_parts = tl.load(
            partial_centered_sum_ptr + partial_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sum_grad = tl.sum(grad_parts, axis=0)
        sum_centered = tl.sum(centered_parts, axis=0)

        invstd = tl.load(arg171_ptr + c).to(tl.float32)
        tl.store(stats_ptr + c, sum_grad * INV_NHW_)
        tl.store(stats_ptr + C_ + c, sum_centered * INV_NHW_ * invstd * invstd)
        tl.store(out1_ptr + c, sum_centered * invstd)

    @triton.jit
    def _write_bn_output_kernel(
        producer_ptr,
        arg170_ptr,
        arg169_ptr,
        arg171_ptr,
        arg4_ptr,
        stats_ptr,
        out0_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n = tl.program_id(1)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        mean = tl.load(arg170_ptr + c).to(tl.float32)
        invstd = tl.load(arg171_ptr + c).to(tl.float32)
        weight = tl.load(arg4_ptr + c).to(tl.float32)
        mean_term = tl.load(stats_ptr + c).to(tl.float32)
        var_term = tl.load(stats_ptr + C_ + c).to(tl.float32)
        output_scale = invstd * weight

        for start in range(0, HW_, BLOCK_HW_):
            hw = start + hw_offsets
            mask = hw < HW_
            offsets = n * C_ * HW_ + c * HW_ + hw

            producer = tl.load(producer_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            activation = tl.load(arg169_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

            centered = activation - mean
            output = (producer - centered * var_term - mean_term) * output_scale
            tl.store(out0_ptr + offsets, output, mask=mask)


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


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        getitem_228,
        sigmoid_15,
        getitem_234,
        add_148,
        add_147,
        arg170_1,
        arg169_1,
        arg171_1,
        arg4_1,
        shape_param_0,
    ) = inputs

    getitem_228 = _require_f32_tensor("getitem_228", getitem_228, OUT0_SHAPE, OUT0_STRIDE)
    sigmoid_15 = _require_f32_tensor("sigmoid_15", sigmoid_15, NC11_SHAPE, NC11_STRIDE)
    getitem_234 = _require_f32_tensor("getitem_234", getitem_234, NC11_SHAPE, NC11_STRIDE)
    add_148 = _require_f32_tensor("add_148", add_148, OUT0_SHAPE, OUT0_STRIDE)
    add_147 = _require_f32_tensor("add_147", add_147, OUT0_SHAPE, OUT0_STRIDE)
    arg170_1 = _require_f32_tensor("arg170_1", arg170_1, (1, C, 1, 1), MEAN_STRIDE)
    arg169_1 = _require_f32_tensor("arg169_1", arg169_1, OUT0_SHAPE, OUT0_STRIDE)
    arg171_1 = _require_f32_tensor("arg171_1", arg171_1, (1, C, 1, 1), MEAN_STRIDE)
    arg4_1 = _require_f32_tensor("arg4_1", arg4_1, OUT1_SHAPE, OUT1_STRIDE)
    _require_shape("_shape_param_0", shape_param_0, OUT0_SHAPE)

    device = getitem_228.device
    if any(
        t.device != device
        for t in (
            sigmoid_15,
            getitem_234,
            add_148,
            add_147,
            arg170_1,
            arg169_1,
            arg171_1,
            arg4_1,
        )
    ):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return (
        getitem_228,
        sigmoid_15,
        getitem_234,
        add_148,
        add_147,
        arg170_1,
        arg169_1,
        arg171_1,
        arg4_1,
    )


def oracle_forward(inputs):
    """Run the exact Repro.forward scope with shared sibling channel reductions."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        getitem_228,
        sigmoid_15,
        getitem_234,
        add_148,
        add_147,
        arg170_1,
        arg169_1,
        arg171_1,
        arg4_1,
    ) = _validate_inputs(inputs)

    partial_sum = torch.empty_strided((C, N), (N, 1), device=getitem_228.device, dtype=torch.float32)
    partial_centered_sum = torch.empty_strided(
        (C, N),
        (N, 1),
        device=getitem_228.device,
        dtype=torch.float32,
    )
    stats = torch.empty_strided((2, C), (C, 1), device=getitem_228.device, dtype=torch.float32)
    producer = torch.empty_strided(OUT0_SHAPE, OUT0_STRIDE, device=getitem_228.device, dtype=torch.float32)
    out0 = torch.empty_strided(OUT0_SHAPE, OUT0_STRIDE, device=getitem_228.device, dtype=torch.float32)
    out1 = torch.empty_strided(OUT1_SHAPE, OUT1_STRIDE, device=getitem_228.device, dtype=torch.float32)

    _partial_bn_sums_kernel[(C, N)](
        getitem_228,
        sigmoid_15,
        getitem_234,
        add_148,
        add_147,
        arg170_1,
        arg169_1,
        producer,
        partial_sum,
        partial_centered_sum,
        C_=C,
        HW_=HW,
        INV_HW_=INV_HW,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
        num_stages=4,
    )
    _finalize_bn_sums_kernel[(C,)](
        partial_sum,
        partial_centered_sum,
        arg171_1,
        stats,
        out1,
        C_=C,
        N_=N,
        INV_NHW_=INV_NHW,
        BLOCK_N_=128,
        num_warps=4,
        num_stages=4,
    )
    _write_bn_output_kernel[(C, N)](
        producer,
        arg170_1,
        arg169_1,
        arg171_1,
        arg4_1,
        stats,
        out0,
        C_=C,
        HW_=HW,
        BLOCK_HW_=BLOCK_HW,
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
