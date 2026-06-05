"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full MobileNetV3 hard-swish plus batch-norm-backward fragment from the original ten repro inputs, sharing the hard-sigmoid/average-pool gradient producer across both `sum([0, 2, 3])` channel reductions and then reusing the finalized channel statistics to emit the returned f32[512,480,14,14] tensor and f32[480] vector; Inductor currently schedules the broadcast pointwise producer, sibling reductions, and dependent BN-backward epilogues as generic reduction/pointwise work rather than one full-scope multi-output plan, so it rereads/recomputes the large producer and cannot coordinate the two accumulators and epilogues; the fix is SCHEDULER_FUSION: teach scheduler/codegen to recognize compatible sibling channel reductions with a shared producer and lower them as a multi-output reduction with fused tensor/vector epilogues."""
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

N = 512
C = 480
H = 14
W = 14
HW = H * W
NHW = N * HW
NUMEL = N * C * HW
INV_HW = 1.0 / HW
INV_NHW = 9.964923469387754e-06

NC11_SHAPE = (N, C, 1, 1)
NC11_STRIDE = (C, 1, 1, 1)
NCHW_SHAPE = (N, C, H, W)
NCHW_STRIDE = (C * HW, HW, W, 1)
PARAM_SHAPE = (1, C, 1, 1)
PARAM_STRIDE = (C, 1, 1, 1)
CHANNEL_SHAPE = (C,)
CHANNEL_STRIDE = (1,)
SCALAR_SHAPE = ()
SCALAR_STRIDE = ()

REDUCE_BLOCK_HW = 256
REDUCE_BLOCK_C = 4
FINALIZE_BLOCK_N = 512
EPILOGUE_BLOCK_ELEMS = 1024


if triton is not None:

    @triton.jit
    def _partial_hardswish_bn_sums_kernel(
        arg254_ptr,
        getitem66_ptr,
        getitem72_ptr,
        add34_ptr,
        full_ptr,
        mean_ptr,
        arg249_ptr,
        partial_where_ptr,
        partial_centered_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        n_idx = tl.program_id(1)
        offs_hw = tl.arange(0, BLOCK_HW_)
        offs_c = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        hw_mask = offs_hw < HW_
        c_mask = offs_c < C_

        tensor_offsets = (
            n_idx * C_ * HW_
            + offs_hw[:, None]
            + offs_c[None, :] * HW_
        )
        channel_offsets = n_idx * C_ + offs_c
        mask = hw_mask[:, None] & c_mask[None, :]

        gate_source = tl.load(arg254_ptr + channel_offsets, mask=c_mask, other=0.0).to(tl.float32)
        pooled_grad = tl.load(getitem72_ptr + channel_offsets, mask=c_mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        gate = tl.minimum(tl.maximum(gate_source + 3.0, 0.0), 6.0) * 0.16666666666666666
        upstream = tl.load(getitem66_ptr + tensor_offsets, mask=mask, other=0.0).to(tl.float32)
        hard_swish_input = tl.load(add34_ptr + tensor_offsets, mask=mask, other=0.0).to(tl.float32)
        base_grad = upstream * gate[None, :] + pooled_grad[None, :] * INV_HW_
        middle_grad = base_grad * (hard_swish_input * 0.3333333333333333 + 0.5)
        grad = tl.where(hard_swish_input < 3.0, middle_grad, base_grad)
        grad = tl.where(hard_swish_input <= -3.0, full_value, grad)
        grad = tl.where(mask, grad, 0.0)

        activation = tl.load(arg249_ptr + tensor_offsets, mask=mask, other=0.0).to(tl.float32)
        centered = activation - mean[None, :]

        partial_offsets = n_idx * C_ + offs_c
        tl.store(partial_where_ptr + partial_offsets, tl.sum(grad, axis=0), mask=c_mask)
        tl.store(partial_centered_ptr + partial_offsets, tl.sum(grad * centered, axis=0), mask=c_mask)

    @triton.jit
    def _finalize_hardswish_bn_sums_kernel(
        partial_where_ptr,
        partial_centered_ptr,
        invstd_ptr,
        weight_ptr,
        stats_ptr,
        out1_ptr,
        C_: tl.constexpr,
        N_: tl.constexpr,
        INV_NHW_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N_)
        offs_c = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        n_mask = offs_n < N_
        c_mask = offs_c < C_
        partial_offsets = offs_n[:, None] * C_ + offs_c[None, :]
        mask = n_mask[:, None] & c_mask[None, :]

        partial_where = tl.load(partial_where_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
        partial_centered = tl.load(partial_centered_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
        sum_where = tl.sum(partial_where, axis=0)
        sum_centered = tl.sum(partial_centered, axis=0)

        invstd = tl.load(invstd_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(stats_ptr + offs_c, sum_where * INV_NHW_, mask=c_mask)
        tl.store(stats_ptr + C_ + offs_c, sum_centered * INV_NHW_ * invstd * invstd, mask=c_mask)
        tl.store(stats_ptr + 2 * C_ + offs_c, invstd * weight, mask=c_mask)
        tl.store(out1_ptr + offs_c, sum_centered * invstd, mask=c_mask)

    @triton.jit
    def _write_hardswish_bn_output_kernel(
        arg254_ptr,
        getitem66_ptr,
        getitem72_ptr,
        add34_ptr,
        full_ptr,
        mean_ptr,
        arg249_ptr,
        stats_ptr,
        out0_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        NUMEL_: tl.constexpr,
        INV_HW_: tl.constexpr,
        BLOCK_ELEMS_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ELEMS_ + tl.arange(0, BLOCK_ELEMS_)
        mask = offsets < NUMEL_
        hw = offsets % HW_
        c = (offsets // HW_) % C_
        n = offsets // (C_ * HW_)
        channel_offsets = n * C_ + c

        gate_source = tl.load(arg254_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        pooled_grad = tl.load(getitem72_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        upstream = tl.load(getitem66_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        hard_swish_input = tl.load(add34_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        gate = tl.minimum(tl.maximum(gate_source + 3.0, 0.0), 6.0) * 0.16666666666666666
        base_grad = upstream * gate + pooled_grad * INV_HW_
        middle_grad = base_grad * (hard_swish_input * 0.3333333333333333 + 0.5)
        grad = tl.where(hard_swish_input < 3.0, middle_grad, base_grad)
        grad = tl.where(hard_swish_input <= -3.0, full_value, grad)

        mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
        activation = tl.load(arg249_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        centered = activation - mean
        mean_term = tl.load(stats_ptr + c, mask=mask, other=0.0).to(tl.float32)
        centered_term = tl.load(stats_ptr + C_ + c, mask=mask, other=0.0).to(tl.float32)
        input_scale = tl.load(stats_ptr + 2 * C_ + c, mask=mask, other=0.0).to(tl.float32)
        out = (grad - centered * centered_term - mean_term) * input_scale
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
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        arg254_1,
        getitem_66,
        getitem_72,
        add_34,
        full,
        arg250_1,
        arg249_1,
        arg251_1,
        arg83_1,
        shape_param_0,
    ) = inputs

    arg254_1 = _require_f32_tensor("arg254_1", arg254_1, NC11_SHAPE, NC11_STRIDE)
    getitem_66 = _require_f32_tensor("getitem_66", getitem_66, NCHW_SHAPE, NCHW_STRIDE)
    getitem_72 = _require_f32_tensor("getitem_72", getitem_72, NC11_SHAPE, NC11_STRIDE)
    add_34 = _require_f32_tensor("add_34", add_34, NCHW_SHAPE, NCHW_STRIDE)
    full = _require_f32_tensor("full", full, SCALAR_SHAPE, SCALAR_STRIDE)
    arg250_1 = _require_f32_tensor("arg250_1", arg250_1, PARAM_SHAPE, PARAM_STRIDE)
    arg249_1 = _require_f32_tensor("arg249_1", arg249_1, NCHW_SHAPE, NCHW_STRIDE)
    arg251_1 = _require_f32_tensor("arg251_1", arg251_1, PARAM_SHAPE, PARAM_STRIDE)
    arg83_1 = _require_f32_tensor("arg83_1", arg83_1, CHANNEL_SHAPE, CHANNEL_STRIDE)

    if list(shape_param_0) != list(NCHW_SHAPE):
        raise ValueError(f"shape_param_0={shape_param_0!r}, expected {list(NCHW_SHAPE)}")

    device = arg254_1.device
    tensors = (getitem_66, getitem_72, add_34, full, arg250_1, arg249_1, arg251_1, arg83_1)
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return arg254_1, getitem_66, getitem_72, add_34, full, arg250_1, arg249_1, arg251_1, arg83_1


def oracle_forward(inputs):
    """Run the full Repro.forward computation with shared channel reductions."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_multi_output_reduction.py")

    (
        arg254_1,
        getitem_66,
        getitem_72,
        add_34,
        full,
        arg250_1,
        arg249_1,
        arg251_1,
        arg83_1,
    ) = _validate_inputs(inputs)

    partial_where = torch.empty_strided((N, C), (C, 1), device=arg254_1.device, dtype=torch.float32)
    partial_centered = torch.empty_strided((N, C), (C, 1), device=arg254_1.device, dtype=torch.float32)
    stats = torch.empty_strided((3, C), (C, 1), device=arg254_1.device, dtype=torch.float32)
    out0 = torch.empty_strided(NCHW_SHAPE, NCHW_STRIDE, device=arg254_1.device, dtype=torch.float32)
    out1 = torch.empty_strided(CHANNEL_SHAPE, CHANNEL_STRIDE, device=arg254_1.device, dtype=torch.float32)

    _partial_hardswish_bn_sums_kernel[(triton.cdiv(C, REDUCE_BLOCK_C), N)](
        arg254_1,
        getitem_66,
        getitem_72,
        add_34,
        full,
        arg250_1,
        arg249_1,
        partial_where,
        partial_centered,
        C_=C,
        HW_=HW,
        INV_HW_=INV_HW,
        BLOCK_HW_=REDUCE_BLOCK_HW,
        BLOCK_C_=REDUCE_BLOCK_C,
        num_warps=4,
        num_stages=4,
    )
    _finalize_hardswish_bn_sums_kernel[(triton.cdiv(C, REDUCE_BLOCK_C),)](
        partial_where,
        partial_centered,
        arg251_1,
        arg83_1,
        stats,
        out1,
        C_=C,
        N_=N,
        INV_NHW_=INV_NHW,
        BLOCK_N_=FINALIZE_BLOCK_N,
        BLOCK_C_=REDUCE_BLOCK_C,
        num_warps=4,
        num_stages=4,
    )
    _write_hardswish_bn_output_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK_ELEMS),)](
        arg254_1,
        getitem_66,
        getitem_72,
        add_34,
        full,
        arg250_1,
        arg249_1,
        stats,
        out0,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        INV_HW_=INV_HW,
        BLOCK_ELEMS_=EPILOGUE_BLOCK_ELEMS,
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
