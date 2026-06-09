"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full MobileNet-style batch-norm-backward algebra by sharing the sibling `sum(add)` and `sum(add * centered)` channel reductions, then uses the finalized scalars to emit both the returned full `[128,24,56,56]` gradient tensor and `[24]` side gradient vector, whereas Inductor schedules the reductions and dependent full-output epilogue as separate graph pieces with extra surrounding pointwise work; Inductor cannot do this today because its scheduler does not form a full-scope multi-output reduction template that shares compatible sibling reductions and sinks their finalized values into a materializing epilogue; the fix is SCHEDULER_FUSION: add scheduler/codegen support for sibling channel reductions with dependent full-output and side-output epilogues in one fused reduction plan."""
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
C = 24
H = 56
W = 56
HW = H * W
NHW = N * HW
INV_NHW = 2.4912308673469386e-06
BLOCK_HW = 4096
BLOCK_N = 128
BLOCK_ELEMS = 1024
OUT0_SHAPE = (N, C, H, W)
OUT0_STRIDE = (C * HW, HW, W, 1)
OUT1_SHAPE = (C,)
OUT1_STRIDE = (1,)
CLASSIFICATION = "SCHEDULER_FUSION"

if triton is not None:

    @triton.jit
    def _partial_sum_sum_kernel(
        getitem_126_ptr,
        getitem_135_ptr,
        arg160_ptr,
        arg347_ptr,
        partial_add_ptr,
        partial_mul_ptr,
        C_: tl.constexpr,
        N_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n = tl.program_id(1)
        hw = tl.arange(0, BLOCK_HW_)
        mask = hw < HW_
        offsets = n * C_ * HW_ + c * HW_ + hw

        add_lhs = tl.load(getitem_126_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add_rhs = tl.load(getitem_135_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(arg160_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(arg347_ptr + c).to(tl.float32)

        add_value = add_lhs + add_rhs
        centered = x - mean
        partial_offset = c * N_ + n
        tl.store(partial_add_ptr + partial_offset, tl.sum(tl.where(mask, add_value, 0.0), axis=0))
        tl.store(
            partial_mul_ptr + partial_offset,
            tl.sum(tl.where(mask, add_value * centered, 0.0), axis=0),
        )

    @triton.jit
    def _finalize_stats_kernel(
        partial_add_ptr,
        partial_mul_ptr,
        arg161_ptr,
        stats_ptr,
        out1_ptr,
        C_: tl.constexpr,
        N_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        inv_nhw: tl.constexpr,
    ):
        c = tl.program_id(0)
        n_offsets = tl.arange(0, BLOCK_N_)
        partial_offsets = c * N_ + n_offsets

        partial_add = tl.load(partial_add_ptr + partial_offsets).to(tl.float32)
        partial_mul = tl.load(partial_mul_ptr + partial_offsets).to(tl.float32)
        sum_add = tl.sum(partial_add, axis=0)
        sum_mul = tl.sum(partial_mul, axis=0)

        arg161 = tl.load(arg161_ptr + c).to(tl.float32)
        tl.store(stats_ptr + c, sum_add * inv_nhw)
        tl.store(stats_ptr + C_ + c, sum_mul * inv_nhw * arg161 * arg161)
        tl.store(out1_ptr + c, sum_mul * arg161)

    @triton.jit
    def _write_full_output_kernel(
        getitem_126_ptr,
        getitem_135_ptr,
        arg160_ptr,
        arg347_ptr,
        arg161_ptr,
        arg16_ptr,
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

        add_lhs = tl.load(getitem_126_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add_rhs = tl.load(getitem_135_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(arg160_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(arg347_ptr + c).to(tl.float32)
        mean_add = tl.load(stats_ptr + c).to(tl.float32)
        correction = tl.load(stats_ptr + C_ + c).to(tl.float32)
        output_scale = tl.load(arg161_ptr + c).to(tl.float32) * tl.load(arg16_ptr + c).to(tl.float32)

        add_value = add_lhs + add_rhs
        centered = x - mean
        out_value = (add_value - centered * correction - mean_add) * output_scale
        tl.store(out0_ptr + offsets, out_value, mask=mask)


def _require_f32_tensor(
    name,
    value,
    shape,
    stride,
):
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
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    getitem_126, getitem_135, arg160_1, arg347_1, arg161_1, arg16_1 = inputs
    getitem_126 = _require_f32_tensor("getitem_126", getitem_126, OUT0_SHAPE, OUT0_STRIDE)
    getitem_135 = _require_f32_tensor("getitem_135", getitem_135, OUT0_SHAPE, OUT0_STRIDE)
    arg160_1 = _require_f32_tensor("arg160_1", arg160_1, OUT0_SHAPE, OUT0_STRIDE)
    arg347_1 = _require_f32_tensor("arg347_1", arg347_1, (1, C, 1, 1), (C, 1, 1, 1))
    arg161_1 = _require_f32_tensor("arg161_1", arg161_1, OUT1_SHAPE, OUT1_STRIDE)
    arg16_1 = _require_f32_tensor("arg16_1", arg16_1, OUT1_SHAPE, OUT1_STRIDE)

    device = getitem_126.device
    if any(t.device != device for t in (getitem_135, arg160_1, arg347_1, arg161_1, arg16_1)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return getitem_126, getitem_135, arg160_1, arg347_1, arg161_1, arg16_1


@oracle_impl(hardware="H100", shapes="(T([128, 24, 56, 56], f32), T([128, 24, 56, 56], f32), T([128, 24, 56, 56], f32), T([1, 24, 1, 1], f32), T([24], f32), T([24], f32))")
def oracle_forward(inputs):
    """Run the exact Repro.forward scope with shared sibling reductions."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_backward_multi_output.py")

    getitem_126, getitem_135, arg160_1, arg347_1, arg161_1, arg16_1 = _validate_inputs(inputs)
    partial_add = torch.empty_strided((C, N), (N, 1), device=getitem_126.device, dtype=torch.float32)
    partial_mul = torch.empty_strided((C, N), (N, 1), device=getitem_126.device, dtype=torch.float32)
    stats = torch.empty_strided((2, C), (C, 1), device=getitem_126.device, dtype=torch.float32)
    out0 = torch.empty_strided(OUT0_SHAPE, OUT0_STRIDE, device=getitem_126.device, dtype=torch.float32)
    out1 = torch.empty_strided(OUT1_SHAPE, OUT1_STRIDE, device=getitem_126.device, dtype=torch.float32)

    _partial_sum_sum_kernel[(C, N)](
        getitem_126,
        getitem_135,
        arg160_1,
        arg347_1,
        partial_add,
        partial_mul,
        C_=C,
        N_=N,
        HW_=HW,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
        num_stages=4,
    )
    _finalize_stats_kernel[(C,)](
        partial_add,
        partial_mul,
        arg161_1,
        stats,
        out1,
        C_=C,
        N_=N,
        BLOCK_N_=BLOCK_N,
        inv_nhw=INV_NHW,
        num_warps=4,
        num_stages=4,
    )
    _write_full_output_kernel[(C, triton.cdiv(NHW, BLOCK_ELEMS))](
        getitem_126,
        getitem_135,
        arg160_1,
        arg347_1,
        arg161_1,
        arg16_1,
        stats,
        out0,
        C_=C,
        HW_=HW,
        NHW_=NHW,
        BLOCK_ELEMS_=BLOCK_ELEMS,
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
