"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 hard-swish batchnorm-backward fragment by sharing the add-plus-hard-swish-gradient producer across both channel reductions, then reusing the finalized per-channel sums to emit both the returned f32[512,16,112,112] input-gradient tensor and f32[16] side gradient vector, whereas Inductor schedules the broadcast hard-swish producer, sibling `sum([0, 2, 3])` reductions, and dependent full-tensor epilogue as generic multi-output reduction work over materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a full-scope template that keeps compatible sibling reductions and their finalized BN-backward epilogue in one fused plan for large NCHW tensors; the fix is SCHEDULER_FUSION: add scheduler/codegen support for hard-swish-gradient channel reductions with dependent tensor and vector epilogues."""
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

N = 512
C = 16
H = 112
W = 112
HW = H * W
NHW = N * HW
INV_NHW = 1.5570192920918366e-07
PARTIAL_BLOCK_HW = 2048
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
    def _partial_hardswish_bn_sums_kernel(
        grad0_ptr,
        grad1_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        partial_where_ptr,
        partial_centered_ptr,
        C_: tl.constexpr,
        N_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n = tl.program_id(1)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        weight = tl.load(weight_ptr + c).to(tl.float32)
        bias = tl.load(bias_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        sum_where = tl.zeros((), dtype=tl.float32)
        sum_centered = tl.zeros((), dtype=tl.float32)

        for start in range(0, HW_, BLOCK_HW_):
            hw = start + hw_offsets
            mask = hw < HW_
            offsets = n * C_ * HW_ + c * HW_ + hw

            grad = (
                tl.load(grad0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                + tl.load(grad1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            )
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            centered = x - mean
            affine = (centered * invstd) * weight + bias
            middle = grad * (affine * 0.3333333333333333 + 0.5)
            where_value = tl.where(affine < 3.0, middle, grad)
            where_value = tl.where(affine <= -3.0, full_value, where_value)
            active = tl.where(mask, where_value, 0.0)

            sum_where += tl.sum(active, axis=0)
            sum_centered += tl.sum(tl.where(mask, where_value * centered, 0.0), axis=0)

        partial_offset = c * N_ + n
        tl.store(partial_where_ptr + partial_offset, sum_where)
        tl.store(partial_centered_ptr + partial_offset, sum_centered)

    @triton.jit
    def _finalize_hardswish_bn_sums_kernel(
        partial_where_ptr,
        partial_centered_ptr,
        invstd_ptr,
        stats_ptr,
        out1_ptr,
        C_: tl.constexpr,
        N_: tl.constexpr,
        INV_NHW_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n_offsets = tl.arange(0, BLOCK_N_)
        mask = n_offsets < N_
        partial_offsets = c * N_ + n_offsets

        partial_where = tl.load(partial_where_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
        partial_centered = tl.load(
            partial_centered_ptr + partial_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sum_where = tl.sum(partial_where, axis=0)
        sum_centered = tl.sum(partial_centered, axis=0)

        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        tl.store(stats_ptr + c, sum_where * INV_NHW_)
        tl.store(stats_ptr + C_ + c, sum_centered * INV_NHW_ * invstd * invstd)
        tl.store(out1_ptr + c, sum_centered * invstd)

    @triton.jit
    def _write_hardswish_bn_output_kernel(
        grad0_ptr,
        grad1_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        stats_ptr,
        out0_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n = tl.program_id(1)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        weight = tl.load(weight_ptr + c).to(tl.float32)
        bias = tl.load(bias_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        mean_term = tl.load(stats_ptr + c).to(tl.float32)
        centered_term = tl.load(stats_ptr + C_ + c).to(tl.float32)
        output_scale = invstd * weight

        for start in range(0, HW_, BLOCK_HW_):
            hw = start + hw_offsets
            mask = hw < HW_
            offsets = n * C_ * HW_ + c * HW_ + hw

            grad = (
                tl.load(grad0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                + tl.load(grad1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            )
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            centered = x - mean
            affine = (centered * invstd) * weight + bias
            middle = grad * (affine * 0.3333333333333333 + 0.5)
            where_value = tl.where(affine < 3.0, middle, grad)
            where_value = tl.where(affine <= -3.0, full_value, where_value)
            output = (where_value - centered * centered_term - mean_term) * output_scale
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


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        getitem_177,
        getitem_183,
        arg134_1,
        arg135_1,
        arg136_1,
        arg2_1,
        arg3_1,
        full,
    ) = inputs

    getitem_177 = _require_f32_tensor("getitem_177", getitem_177, OUT0_SHAPE, OUT0_STRIDE)
    getitem_183 = _require_f32_tensor("getitem_183", getitem_183, OUT0_SHAPE, OUT0_STRIDE)
    arg134_1 = _require_f32_tensor("arg134_1", arg134_1, OUT0_SHAPE, OUT0_STRIDE)
    arg135_1 = _require_f32_tensor("arg135_1", arg135_1, PARAM_SHAPE, PARAM_STRIDE)
    arg136_1 = _require_f32_tensor("arg136_1", arg136_1, PARAM_SHAPE, PARAM_STRIDE)
    arg2_1 = _require_f32_tensor("arg2_1", arg2_1, CHANNEL_SHAPE, CHANNEL_STRIDE)
    arg3_1 = _require_f32_tensor("arg3_1", arg3_1, CHANNEL_SHAPE, CHANNEL_STRIDE)
    full = _require_f32_tensor("full", full, SCALAR_SHAPE, SCALAR_STRIDE)

    device = getitem_177.device
    if any(t.device != device for t in (getitem_183, arg134_1, arg135_1, arg136_1, arg2_1, arg3_1, full)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return getitem_177, getitem_183, arg134_1, arg135_1, arg136_1, arg2_1, arg3_1, full


@oracle_impl(hardware="H100", shapes="(T([512, 16, 112, 112], f32), T([512, 16, 112, 112], f32), T([512, 16, 112, 112], f32), T([1, 16, 1, 1], f32), T([1, 16, 1, 1], f32), T([16], f32), T([16], f32), T([], f32))")
def oracle_forward(inputs):
    """Run the exact Repro.forward scope with shared hard-swish BN-backward reductions."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_hardswish_backward_full_scope.py")

    (
        getitem_177,
        getitem_183,
        arg134_1,
        arg135_1,
        arg136_1,
        arg2_1,
        arg3_1,
        full,
    ) = _validate_inputs(inputs)

    partial_where = torch.empty_strided((C, N), (N, 1), device=getitem_177.device, dtype=torch.float32)
    partial_centered = torch.empty_strided((C, N), (N, 1), device=getitem_177.device, dtype=torch.float32)
    stats = torch.empty_strided((2, C), (C, 1), device=getitem_177.device, dtype=torch.float32)
    out0 = torch.empty_strided(OUT0_SHAPE, OUT0_STRIDE, device=getitem_177.device, dtype=torch.float32)
    out1 = torch.empty_strided(CHANNEL_SHAPE, CHANNEL_STRIDE, device=getitem_177.device, dtype=torch.float32)

    _partial_hardswish_bn_sums_kernel[(C, N)](
        getitem_177,
        getitem_183,
        arg134_1,
        arg135_1,
        arg136_1,
        arg2_1,
        arg3_1,
        full,
        partial_where,
        partial_centered,
        C_=C,
        N_=N,
        HW_=HW,
        BLOCK_HW_=PARTIAL_BLOCK_HW,
        num_warps=8,
        num_stages=4,
    )
    _finalize_hardswish_bn_sums_kernel[(C,)](
        partial_where,
        partial_centered,
        arg136_1,
        stats,
        out1,
        C_=C,
        N_=N,
        INV_NHW_=INV_NHW,
        BLOCK_N_=N,
        num_warps=8,
        num_stages=4,
    )
    _write_hardswish_bn_output_kernel[(C, N)](
        getitem_177,
        getitem_183,
        arg134_1,
        arg135_1,
        arg136_1,
        arg2_1,
        arg3_1,
        full,
        stats,
        out0,
        C_=C,
        HW_=HW,
        BLOCK_HW_=OUT_BLOCK_HW,
        num_warps=4,
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
