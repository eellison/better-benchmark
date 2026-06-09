"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DCGAN batchnorm-backward leaky-ReLU scope from Repro.forward, including both shared channel reductions, the dependent full tensor epilogue, and the scale-gradient vector, using a split-K multi-accumulator Triton reduction followed by exact f32 scalar finalization and a dense epilogue writer; Inductor's measured full-scope output is already within the CUDAGraph harness floor for the same required reads, f32 channel reductions, coefficient finalization, and dense output store rather than exposing a subset-reduction-only scope mismatch; Inductor cannot materially improve this repro with a local scheduler-fusion, scatter-reduce, cooperative split-K, algebraic-elimination, or recompute change because the full computation is dominated by mandatory dense tensor traffic and exact f32 reduction/epilogue work; the fix is BANDWIDTH_BOUND: record this oracle as at-floor unless broader memory-traffic or reduction-codegen improvements move both implementations."""
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

N = 1024
C = 128
H = 16
W = 16
HW = H * W
REDUCE_N = N * HW
TOTAL_N = N * C * HW
SCALE = 3.814697265625e-06
REDUCE_BLOCK = 8192
NUM_REDUCE_BLOCKS = REDUCE_N // REDUCE_BLOCK
EPILOGUE_BLOCK = 1024

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

if triton is not None:

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
    def _dual_summary_partials_kernel(
        mask_src,
        grad_src,
        x_src,
        mean_src,
        partial_where,
        partial_mul,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        REDUCE_N_: tl.constexpr,
        NUM_REDUCE_BLOCKS_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0)
        block = tl.program_id(1)
        k = block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = k < REDUCE_N_
        n = k // HW_
        hw = k - n * HW_
        offsets = n * (C_ * HW_) + c * HW_ + hw

        mask_val = tl.load(mask_src + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_val = tl.load(grad_src + offsets, mask=mask, other=0.0).to(tl.float32)
        x_val = tl.load(x_src + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_val = tl.load(mean_src + c).to(tl.float32)

        leaky = _mul_rn_f32(grad_val, 0.2)
        selected = tl.where(mask_val > 0.0, grad_val, leaky)
        centered = _sub_rn_f32(x_val, mean_val)
        mul_val = _mul_rn_f32(selected, centered)

        sum_where = tl.sum(tl.where(mask, selected, 0.0), axis=0)
        sum_mul = tl.sum(tl.where(mask, mul_val, 0.0), axis=0)
        out_offset = c * NUM_REDUCE_BLOCKS_ + block
        tl.store(partial_where + out_offset, sum_where)
        tl.store(partial_mul + out_offset, sum_mul)

    @triton.jit
    def _finalize_channel_scalars_kernel(
        partial_where,
        partial_mul,
        invstd_src,
        weight_src,
        coeff_where,
        coeff_mul,
        coeff_weight,
        out1,
        NUM_REDUCE_BLOCKS_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        c = tl.program_id(0)
        b = tl.arange(0, BLOCK_B)
        mask = b < NUM_REDUCE_BLOCKS_
        offsets = c * NUM_REDUCE_BLOCKS_ + b

        sum_where = tl.sum(tl.load(partial_where + offsets, mask=mask, other=0.0), axis=0)
        sum_mul = tl.sum(tl.load(partial_mul + offsets, mask=mask, other=0.0), axis=0)
        invstd = tl.load(invstd_src + c).to(tl.float32)
        weight = tl.load(weight_src + c).to(tl.float32)

        sum_where_scaled = _mul_rn_f32(sum_where, SCALE_)
        sum_mul_scaled = _mul_rn_f32(sum_mul, SCALE_)
        invstd_sq = _mul_rn_f32(invstd, invstd)
        mul_coeff = _mul_rn_f32(sum_mul_scaled, invstd_sq)
        out_weight = _mul_rn_f32(invstd, weight)
        grad_weight = _mul_rn_f32(sum_mul, invstd)

        tl.store(coeff_where + c, sum_where_scaled)
        tl.store(coeff_mul + c, mul_coeff)
        tl.store(coeff_weight + c, out_weight)
        tl.store(out1 + c, grad_weight)

    @triton.jit
    def _dense_epilogue_kernel(
        mask_src,
        grad_src,
        x_src,
        mean_src,
        coeff_where,
        coeff_mul,
        coeff_weight,
        out0,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        TOTAL_N_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL_N_
        c = (offsets // HW_) % C_

        mask_val = tl.load(mask_src + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_val = tl.load(grad_src + offsets, mask=mask, other=0.0).to(tl.float32)
        x_val = tl.load(x_src + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_val = tl.load(mean_src + c, mask=mask, other=0.0).to(tl.float32)
        where_coeff = tl.load(coeff_where + c, mask=mask, other=0.0).to(tl.float32)
        mul_coeff = tl.load(coeff_mul + c, mask=mask, other=0.0).to(tl.float32)
        out_weight = tl.load(coeff_weight + c, mask=mask, other=0.0).to(tl.float32)

        leaky = _mul_rn_f32(grad_val, 0.2)
        selected = tl.where(mask_val > 0.0, grad_val, leaky)
        centered = _sub_rn_f32(x_val, mean_val)
        correction = _mul_rn_f32(centered, mul_coeff)
        tmp = _sub_rn_f32(selected, correction)
        tmp = _sub_rn_f32(tmp, where_coeff)
        out = _mul_rn_f32(tmp, out_weight)
        tl.store(out0 + offsets, out, mask=mask)


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 6:
        raise ValueError(f"expected 6 repro inputs, got {len(inputs)}")
    if not all(isinstance(x, torch.Tensor) for x in inputs):
        raise TypeError("all repro inputs must be tensors")

    arg12_1, getitem_6, arg10_1, arg22_1, arg11_1, arg3_1 = inputs
    expected_big = (N, C, H, W)
    if tuple(arg12_1.shape) != expected_big:
        raise ValueError(f"unexpected arg12_1 shape {tuple(arg12_1.shape)}")
    if tuple(getitem_6.shape) != expected_big or tuple(arg10_1.shape) != expected_big:
        raise ValueError("unexpected dense input shape")
    if tuple(arg22_1.shape) != (1, C, 1, 1):
        raise ValueError(f"unexpected arg22_1 shape {tuple(arg22_1.shape)}")
    if tuple(arg11_1.shape) != (C,) or tuple(arg3_1.shape) != (C,):
        raise ValueError("unexpected channel vector shape")

    tensors = (arg12_1, getitem_6, arg10_1, arg22_1, arg11_1, arg3_1)
    device = arg12_1.device
    if device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("this oracle expects the captured f32 repro inputs")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("this oracle expects the captured contiguous input layouts")

    return tensors


@oracle_impl(hardware="H100", shapes="(T([1024, 128, 16, 16], f32), T([1024, 128, 16, 16], f32), T([1024, 128, 16, 16], f32), T([1, 128, 1, 1], f32), T([128], f32), T([128], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward-equivalent oracle computation."""
    arg12_1, getitem_6, arg10_1, arg22_1, arg11_1, arg3_1 = _validate_inputs(inputs)
    device = arg12_1.device

    partial_where = torch.empty_strided(
        (C, NUM_REDUCE_BLOCKS),
        (NUM_REDUCE_BLOCKS, 1),
        device=device,
        dtype=torch.float32,
    )
    partial_mul = torch.empty_strided(
        (C, NUM_REDUCE_BLOCKS),
        (NUM_REDUCE_BLOCKS, 1),
        device=device,
        dtype=torch.float32,
    )
    coeff_where = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    coeff_mul = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    coeff_weight = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    out0 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.float32,
    )
    out1 = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)

    _dual_summary_partials_kernel[(C, NUM_REDUCE_BLOCKS)](
        arg12_1,
        getitem_6,
        arg10_1,
        arg22_1,
        partial_where,
        partial_mul,
        C_=C,
        HW_=HW,
        REDUCE_N_=REDUCE_N,
        NUM_REDUCE_BLOCKS_=NUM_REDUCE_BLOCKS,
        BLOCK_K=REDUCE_BLOCK,
        num_warps=8,
    )
    _finalize_channel_scalars_kernel[(C,)](
        partial_where,
        partial_mul,
        arg11_1,
        arg3_1,
        coeff_where,
        coeff_mul,
        coeff_weight,
        out1,
        NUM_REDUCE_BLOCKS_=NUM_REDUCE_BLOCKS,
        SCALE_=SCALE,
        BLOCK_B=NUM_REDUCE_BLOCKS,
        num_warps=1,
    )
    _dense_epilogue_kernel[(triton.cdiv(TOTAL_N, EPILOGUE_BLOCK),)](
        arg12_1,
        getitem_6,
        arg10_1,
        arg22_1,
        coeff_where,
        coeff_mul,
        coeff_weight,
        out0,
        C_=C,
        HW_=HW,
        TOTAL_N_=TOTAL_N,
        BLOCK_N=EPILOGUE_BLOCK,
        num_warps=4,
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
