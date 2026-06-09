"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full RepVGG BN-backward-style captured scope with one shared masked producer, one three-accumulator channel reduction (`sum(where)`, `sum(where * centered1)`, and `sum(where * centered2)`), both returned scale-gradient vectors, and both dense BN epilogues; the measured full-scope oracle is only about 1.05x ahead of compiled Inductor because both paths are dominated by the required large activation reads and two full dense output stores; Inductor cannot materially improve this case through a narrower scheduler-only change because the remaining gap is within measurement noise and memory-traffic floor for the reduction-plus-two-epilogue computation; the fix is BANDWIDTH_BOUND: record this as at floor unless a broader memory-traffic or launch-overhead improvement moves both oracle and compiled paths."""
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


N = 128
C = 64
H = 112
W = 112
HW = H * W
NHW = N * HW
TOTAL = N * C * HW
INV_NHW = 6.228077168367346e-07
REDUCE_BLOCK = 1024
NUM_REDUCE_BLOCKS = (NHW + REDUCE_BLOCK - 1) // REDUCE_BLOCK
FINAL_BLOCK = 2048
EPILOGUE_BLOCK = 1024

ACT_SHAPE = (N, C, H, W)
ACT_STRIDE = (C * HW, HW, W, 1)
MEAN_SHAPE = (1, C, 1, 1)
MEAN_STRIDE = (C, 1, 1, 1)
VEC_SHAPE = (C,)
VEC_STRIDE = (1,)
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
    def _partial_reductions_kernel(
        getitem_120_ptr,
        getitem_123_ptr,
        arg113_ptr,
        full_ptr,
        arg111_ptr,
        arg295_ptr,
        arg109_ptr,
        arg296_ptr,
        partial_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        NHW_: tl.constexpr,
        NUM_REDUCE_BLOCKS_: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        c = tl.program_id(0)
        block = tl.program_id(1)
        r = block * BLOCK_R + tl.arange(0, BLOCK_R)
        active = r < NHW_
        n = r // HW_
        hw = r - n * HW_
        offset = n * (C_ * HW_) + c * HW_ + hw

        lhs = tl.load(getitem_120_ptr + offset, mask=active, other=0.0).to(tl.float32)
        rhs = tl.load(getitem_123_ptr + offset, mask=active, other=0.0).to(tl.float32)
        mask_value = tl.load(arg113_ptr + offset, mask=active, other=0.0).to(tl.float32)
        x1 = tl.load(arg111_ptr + offset, mask=active, other=0.0).to(tl.float32)
        x2 = tl.load(arg109_ptr + offset, mask=active, other=0.0).to(tl.float32)
        mean1 = tl.load(arg295_ptr + c).to(tl.float32)
        mean2 = tl.load(arg296_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        add_value = _add_rn_f32(lhs, rhs)
        where_value = tl.where(mask_value <= 0.0, full_value, add_value)
        centered1 = _sub_rn_f32(x1, mean1)
        centered2 = _sub_rn_f32(x2, mean2)
        product1 = _mul_rn_f32(where_value, centered1)
        product2 = _mul_rn_f32(where_value, centered2)

        partial_offset = c * NUM_REDUCE_BLOCKS_ + block
        plane_stride = C_ * NUM_REDUCE_BLOCKS_
        zeros = tl.full((BLOCK_R,), 0.0, tl.float32)
        tl.store(
            partial_ptr + partial_offset,
            tl.sum(tl.where(active, where_value, zeros), axis=0),
        )
        tl.store(
            partial_ptr + plane_stride + partial_offset,
            tl.sum(tl.where(active, product1, zeros), axis=0),
        )
        tl.store(
            partial_ptr + plane_stride * 2 + partial_offset,
            tl.sum(tl.where(active, product2, zeros), axis=0),
        )

    @triton.jit
    def _finalize_kernel(
        partial_ptr,
        arg112_ptr,
        arg4_ptr,
        arg110_ptr,
        arg2_ptr,
        stats_ptr,
        out_vec1_ptr,
        out_vec2_ptr,
        C_: tl.constexpr,
        NUM_REDUCE_BLOCKS_: tl.constexpr,
        BLOCK_B: tl.constexpr,
        INV_NHW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        b = tl.arange(0, BLOCK_B)
        mask = b < NUM_REDUCE_BLOCKS_
        partial_offset = c * NUM_REDUCE_BLOCKS_ + b
        plane_stride = C_ * NUM_REDUCE_BLOCKS_

        sum_where = tl.sum(
            tl.load(partial_ptr + partial_offset, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        sum_product1 = tl.sum(
            tl.load(partial_ptr + plane_stride + partial_offset, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        sum_product2 = tl.sum(
            tl.load(partial_ptr + plane_stride * 2 + partial_offset, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        invstd1 = tl.load(arg112_ptr + c).to(tl.float32)
        weight1 = tl.load(arg4_ptr + c).to(tl.float32)
        invstd2 = tl.load(arg110_ptr + c).to(tl.float32)
        weight2 = tl.load(arg2_ptr + c).to(tl.float32)

        mean_where = _mul_rn_f32(sum_where, INV_NHW_)
        mean_product1 = _mul_rn_f32(sum_product1, INV_NHW_)
        invstd1_sq = _mul_rn_f32(invstd1, invstd1)
        coeff1 = _mul_rn_f32(mean_product1, invstd1_sq)
        weight_scale1 = _mul_rn_f32(invstd1, weight1)
        out_vec1 = _mul_rn_f32(sum_product1, invstd1)

        mean_product2 = _mul_rn_f32(sum_product2, INV_NHW_)
        invstd2_sq = _mul_rn_f32(invstd2, invstd2)
        coeff2 = _mul_rn_f32(mean_product2, invstd2_sq)
        weight_scale2 = _mul_rn_f32(invstd2, weight2)
        out_vec2 = _mul_rn_f32(sum_product2, invstd2)

        tl.store(stats_ptr + c, mean_where)
        tl.store(stats_ptr + C_ + c, coeff1)
        tl.store(stats_ptr + C_ * 2 + c, weight_scale1)
        tl.store(stats_ptr + C_ * 3 + c, coeff2)
        tl.store(stats_ptr + C_ * 4 + c, weight_scale2)
        tl.store(out_vec1_ptr + c, out_vec1)
        tl.store(out_vec2_ptr + c, out_vec2)

    @triton.jit
    def _dense_epilogue_kernel(
        getitem_120_ptr,
        getitem_123_ptr,
        arg113_ptr,
        full_ptr,
        arg111_ptr,
        arg295_ptr,
        arg109_ptr,
        arg296_ptr,
        stats_ptr,
        out_dense1_ptr,
        out_dense2_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        TOTAL_: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        active = offsets < TOTAL_
        c = (offsets // HW_) % C_

        lhs = tl.load(getitem_120_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        rhs = tl.load(getitem_123_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        mask_value = tl.load(arg113_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        x1 = tl.load(arg111_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        x2 = tl.load(arg109_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        mean1 = tl.load(arg295_ptr + c, mask=active, other=0.0).to(tl.float32)
        mean2 = tl.load(arg296_ptr + c, mask=active, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        mean_where = tl.load(stats_ptr + c, mask=active, other=0.0).to(tl.float32)
        coeff1 = tl.load(stats_ptr + C_ + c, mask=active, other=0.0).to(tl.float32)
        weight_scale1 = tl.load(stats_ptr + C_ * 2 + c, mask=active, other=0.0).to(tl.float32)
        coeff2 = tl.load(stats_ptr + C_ * 3 + c, mask=active, other=0.0).to(tl.float32)
        weight_scale2 = tl.load(stats_ptr + C_ * 4 + c, mask=active, other=0.0).to(tl.float32)

        add_value = _add_rn_f32(lhs, rhs)
        where_value = tl.where(mask_value <= 0.0, full_value, add_value)

        centered1 = _sub_rn_f32(x1, mean1)
        correction1 = _mul_rn_f32(centered1, coeff1)
        tmp1 = _sub_rn_f32(where_value, correction1)
        tmp1 = _sub_rn_f32(tmp1, mean_where)
        out1 = _mul_rn_f32(tmp1, weight_scale1)

        centered2 = _sub_rn_f32(x2, mean2)
        correction2 = _mul_rn_f32(centered2, coeff2)
        tmp2 = _sub_rn_f32(where_value, correction2)
        tmp2 = _sub_rn_f32(tmp2, mean_where)
        out2 = _mul_rn_f32(tmp2, weight_scale2)

        tl.store(out_dense1_ptr + offsets, out1, mask=active)
        tl.store(out_dense2_ptr + offsets, out2, mask=active)


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
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    (
        getitem_120,
        getitem_123,
        arg113_1,
        full_1,
        arg111_1,
        arg295_1,
        arg112_1,
        arg4_1,
        arg109_1,
        arg296_1,
        arg110_1,
        arg2_1,
    ) = inputs

    getitem_120 = _require_f32_tensor("getitem_120", getitem_120, ACT_SHAPE, ACT_STRIDE)
    getitem_123 = _require_f32_tensor("getitem_123", getitem_123, ACT_SHAPE, ACT_STRIDE)
    arg113_1 = _require_f32_tensor("arg113_1", arg113_1, ACT_SHAPE, ACT_STRIDE)
    full_1 = _require_f32_tensor("full_1", full_1, SCALAR_SHAPE, SCALAR_STRIDE)
    arg111_1 = _require_f32_tensor("arg111_1", arg111_1, ACT_SHAPE, ACT_STRIDE)
    arg295_1 = _require_f32_tensor("arg295_1", arg295_1, MEAN_SHAPE, MEAN_STRIDE)
    arg112_1 = _require_f32_tensor("arg112_1", arg112_1, VEC_SHAPE, VEC_STRIDE)
    arg4_1 = _require_f32_tensor("arg4_1", arg4_1, VEC_SHAPE, VEC_STRIDE)
    arg109_1 = _require_f32_tensor("arg109_1", arg109_1, ACT_SHAPE, ACT_STRIDE)
    arg296_1 = _require_f32_tensor("arg296_1", arg296_1, MEAN_SHAPE, MEAN_STRIDE)
    arg110_1 = _require_f32_tensor("arg110_1", arg110_1, VEC_SHAPE, VEC_STRIDE)
    arg2_1 = _require_f32_tensor("arg2_1", arg2_1, VEC_SHAPE, VEC_STRIDE)

    device = getitem_120.device
    if any(
        tensor.device != device
        for tensor in (
            getitem_123,
            arg113_1,
            full_1,
            arg111_1,
            arg295_1,
            arg112_1,
            arg4_1,
            arg109_1,
            arg296_1,
            arg110_1,
            arg2_1,
        )
    ):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return (
        getitem_120,
        getitem_123,
        arg113_1,
        full_1,
        arg111_1,
        arg295_1,
        arg112_1,
        arg4_1,
        arg109_1,
        arg296_1,
        arg110_1,
        arg2_1,
    )


@oracle_impl(hardware="H100", shapes="(T([128, 64, 112, 112], f32), T([128, 64, 112, 112], f32), T([128, 64, 112, 112], f32), T([], f32), T([128, 64, 112, 112], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([128, 64, 112, 112], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32))")
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
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        getitem_120,
        getitem_123,
        arg113_1,
        full_1,
        arg111_1,
        arg295_1,
        arg112_1,
        arg4_1,
        arg109_1,
        arg296_1,
        arg110_1,
        arg2_1,
    ) = _validate_inputs(inputs)

    device = getitem_120.device
    partial = torch.empty((3, C, NUM_REDUCE_BLOCKS), device=device, dtype=torch.float32)
    stats = torch.empty((5, C), device=device, dtype=torch.float32)
    out_dense1 = torch.empty_strided(ACT_SHAPE, ACT_STRIDE, device=device, dtype=torch.float32)
    out_vec1 = torch.empty_strided(VEC_SHAPE, VEC_STRIDE, device=device, dtype=torch.float32)
    out_dense2 = torch.empty_strided(ACT_SHAPE, ACT_STRIDE, device=device, dtype=torch.float32)
    out_vec2 = torch.empty_strided(VEC_SHAPE, VEC_STRIDE, device=device, dtype=torch.float32)

    _partial_reductions_kernel[(C, NUM_REDUCE_BLOCKS)](
        getitem_120,
        getitem_123,
        arg113_1,
        full_1,
        arg111_1,
        arg295_1,
        arg109_1,
        arg296_1,
        partial,
        C_=C,
        HW_=HW,
        NHW_=NHW,
        NUM_REDUCE_BLOCKS_=NUM_REDUCE_BLOCKS,
        BLOCK_R=REDUCE_BLOCK,
        num_warps=8,
    )
    _finalize_kernel[(C,)](
        partial,
        arg112_1,
        arg4_1,
        arg110_1,
        arg2_1,
        stats,
        out_vec1,
        out_vec2,
        C_=C,
        NUM_REDUCE_BLOCKS_=NUM_REDUCE_BLOCKS,
        BLOCK_B=FINAL_BLOCK,
        INV_NHW_=INV_NHW,
        num_warps=8,
    )
    _dense_epilogue_kernel[(triton.cdiv(TOTAL, EPILOGUE_BLOCK),)](
        getitem_120,
        getitem_123,
        arg113_1,
        full_1,
        arg111_1,
        arg295_1,
        arg109_1,
        arg296_1,
        stats,
        out_dense1,
        out_dense2,
        C_=C,
        HW_=HW,
        TOTAL_=TOTAL,
        BLOCK_E=EPILOGUE_BLOCK,
        num_warps=8,
    )

    return (out_dense1, out_vec1, out_dense2, out_vec2)


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
