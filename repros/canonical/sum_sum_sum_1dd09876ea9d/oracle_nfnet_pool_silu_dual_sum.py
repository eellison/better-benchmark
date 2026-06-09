"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet pooled-SiLU backward fragment returned by `Repro.forward`, including the fixed 2x2 avg-pool backward expansion, exact captured sigmoid, scalar all-element reduction, per-`(N,C)` spatial reduction, sigmoid-derivative gate, and final channel reduction; Inductor currently treats the structured pool producer, the scalar sibling sum, and the gated channel sum as separate generic regions even though they share the same producer and reduction domain; Inductor cannot do this today because the scheduler does not co-plan sibling reductions where one output is a global scalar and the other is a channel reduction with a per-`(N,C)` gate applied after the spatial sum; the fix is SCHEDULER_FUSION: fuse the structured pool producer and shared pointwise chain into one multi-output reduction template with a channel finalizer while preserving exact f32 operation order and sigmoid numerics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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


N = 128
C = 512
H = 24
W = 24
HW = H * W
POOL_H = 12
POOL_W = 12
NC = N * C
FINAL_BLOCK_N = 128
FINAL_BLOCK_C = 1024
SPATIAL_BLOCK = 1024

MAIN_SHAPE = (N, C, H, W)
MAIN_STRIDE = (C * HW, HW, W, 1)
POOL_SHAPE = (N, C, POOL_H, POOL_W)
POOL_STRIDE = (C * POOL_H * POOL_W, POOL_H * POOL_W, POOL_W, 1)
GATE_SHAPE = (N, C, 1, 1)
GATE_STRIDE = (C, 1, 1, 1)
SCALAR_SHAPE = ()
SCALAR_STRIDE = ()


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _add_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sub_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _nc_spatial_reduce_kernel(
        pool_grad_ptr,
        getitem_165_ptr,
        arg460_ptr,
        gate_input_ptr,
        arg227_ptr,
        arg44_ptr,
        scalar_nc_ptr,
        vector_nc_ptr,
        HW_: tl.constexpr,
        H_: tl.constexpr,
        W_: tl.constexpr,
        POOL_H_: tl.constexpr,
        POOL_W_: tl.constexpr,
        C_: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        nc = tl.program_id(0)
        n = nc // C_
        c = nc - n * C_
        hw = tl.arange(0, BLOCK_HW)
        mask = hw < HW_
        h = hw // W_
        w = hw - h * W_

        main_offset = n * (C_ * HW_) + c * HW_ + hw
        pool_offset = n * (C_ * POOL_H_ * POOL_W_) + c * (POOL_H_ * POOL_W_) + (h // 2) * POOL_W_ + (w // 2)

        pool_value = tl.load(pool_grad_ptr + pool_offset, mask=mask, other=0.0).to(tl.float32)
        getitem_value = tl.load(getitem_165_ptr + main_offset, mask=mask, other=0.0).to(tl.float32)
        arg460_value = tl.load(arg460_ptr + main_offset, mask=mask, other=0.0).to(tl.float32)
        arg227_value = tl.load(arg227_ptr + main_offset, mask=mask, other=0.0).to(tl.float32)
        arg44_value = tl.load(arg44_ptr).to(tl.float32)
        gate_x = tl.load(gate_input_ptr + nc).to(tl.float32)

        pool_expanded = _mul_rn_f32(pool_value, 0.25)
        add_tensor = _add_rn_f32(getitem_value, pool_expanded)
        mul0 = _mul_rn_f32(add_tensor, 0.9622504486493761)
        mul1 = _mul_rn_f32(mul0, 1.7015043497085571)
        mul2 = _mul_rn_f32(mul1, arg460_value)
        base = _mul_rn_f32(mul2, 0.2)

        sigmoid = tl.sigmoid(gate_x)

        scalar_rhs = _mul_rn_f32(arg227_value, sigmoid)
        scalar_rhs = _mul_rn_f32(scalar_rhs, 2.0)
        scalar_values = _mul_rn_f32(base, scalar_rhs)

        vector_values = _mul_rn_f32(base, arg44_value)
        vector_values = _mul_rn_f32(vector_values, 2.0)
        vector_values = _mul_rn_f32(vector_values, arg227_value)

        scalar_sum = tl.sum(tl.where(mask, scalar_values, 0.0), axis=0)
        spatial_sum = tl.sum(tl.where(mask, vector_values, 0.0), axis=0)
        one_minus = _sub_rn_f32(1.0, sigmoid)
        sigmoid_grad = _mul_rn_f32(sigmoid, one_minus)
        vector_value = _mul_rn_f32(spatial_sum, sigmoid_grad)

        tl.store(scalar_nc_ptr + nc, scalar_sum)
        tl.store(vector_nc_ptr + nc, vector_value)

    @triton.jit
    def _channel_finalize_kernel(
        scalar_nc_ptr,
        vector_nc_ptr,
        channel_scalar_ptr,
        out_vec_ptr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        c = tl.program_id(0)
        n = tl.arange(0, BLOCK_N)
        mask = n < N_
        offsets = n * C_ + c
        scalar_values = tl.load(scalar_nc_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        vector_values = tl.load(vector_nc_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        scalar_sum = tl.sum(scalar_values, axis=0)
        vector_sum = tl.sum(vector_values, axis=0)
        tl.store(channel_scalar_ptr + c, scalar_sum)
        tl.store(out_vec_ptr + c, vector_sum)

    @triton.jit
    def _scalar_finalize_kernel(
        channel_scalar_ptr,
        out_scalar_ptr,
        C_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c = tl.arange(0, BLOCK_C)
        mask = c < C_
        values = tl.load(channel_scalar_ptr + c, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_scalar_ptr, tl.sum(values, axis=0))


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} stride {tuple(value.stride())} does not match {stride}")
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype {value.dtype} does not match {dtype}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have storage_offset=0, got {value.storage_offset()}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")
    pool_grad, pool_shape_ref, getitem_165, arg460, gate_input, arg227, arg44 = inputs
    pool_grad = _expect_tensor("getitem_168", pool_grad, POOL_SHAPE, POOL_STRIDE, torch.float32)
    _expect_tensor("arg231_1", pool_shape_ref, MAIN_SHAPE, MAIN_STRIDE, torch.float32)
    getitem_165 = _expect_tensor("getitem_165", getitem_165, MAIN_SHAPE, MAIN_STRIDE, torch.float32)
    arg460 = _expect_tensor("arg460_1", arg460, MAIN_SHAPE, MAIN_STRIDE, torch.float32)
    gate_input = _expect_tensor("arg230_1", gate_input, GATE_SHAPE, GATE_STRIDE, torch.float32)
    arg227 = _expect_tensor("arg227_1", arg227, MAIN_SHAPE, MAIN_STRIDE, torch.float32)
    arg44 = _expect_tensor("arg44_1", arg44, SCALAR_SHAPE, SCALAR_STRIDE, torch.float32)
    return pool_grad, getitem_165, arg460, gate_input, arg227, arg44


@oracle_impl(hardware="H100", shapes="(T([128, 512, 12, 12], f32), T([128, 512, 24, 24], f32), T([128, 512, 24, 24], f32), T([128, 512, 24, 24], f32), T([128, 512, 1, 1], f32), T([128, 512, 24, 24], f32), T([], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation for the canonical NFNet shape."""
    pool_grad, getitem_165, arg460, gate_input, arg227, arg44 = _validate_inputs(inputs)

    scalar_nc = torch.empty((NC,), device=getitem_165.device, dtype=torch.float32)
    vector_nc = torch.empty_like(scalar_nc)
    channel_scalar = torch.empty((C,), device=getitem_165.device, dtype=torch.float32)
    out_vec = torch.empty((C,), device=getitem_165.device, dtype=torch.float32)
    out_scalar = torch.empty((), device=getitem_165.device, dtype=torch.float32)

    _nc_spatial_reduce_kernel[(NC,)](
        pool_grad,
        getitem_165,
        arg460,
        gate_input,
        arg227,
        arg44,
        scalar_nc,
        vector_nc,
        HW_=HW,
        H_=H,
        W_=W,
        POOL_H_=POOL_H,
        POOL_W_=POOL_W,
        C_=C,
        BLOCK_HW=SPATIAL_BLOCK,
        num_warps=4,
        num_stages=2,
    )
    _channel_finalize_kernel[(C,)](
        scalar_nc,
        vector_nc,
        channel_scalar,
        out_vec,
        N_=N,
        C_=C,
        BLOCK_N=FINAL_BLOCK_N,
        num_warps=4,
        num_stages=1,
    )
    _scalar_finalize_kernel[(1,)](
        channel_scalar,
        out_scalar,
        C_=C,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=1,
    )

    return out_scalar, out_vec


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
