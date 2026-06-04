"""
Full-scope oracle for pointwise_482b0c1952cd.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full
Lennard-Jones broadcast pointwise result returned by Repro.forward,
`tanh(arg2[:, 0:1] * arg0[:, 0].unsqueeze(0) + arg1)`, over the complete
`float32[1000, 16]` output in one hand-specialized Triton kernel. It skips the
layout-only permute and the two multiply-by-one nodes by loading the broadcast
row and column vectors directly, whereas Inductor already lowers the decomposed
permute/mul/mul/add/tanh graph to one fused pointwise Triton kernel. Inductor
cannot materially do less required work here because every output element must
perform one multiply, one add, one tanh, and one fresh contiguous store; the
remaining difference is kernel-template specialization around a mandatory
elementwise materialization. The fix classification is BANDWIDTH_BOUND: no
scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, recompute, or
new-pattern Inductor change is indicated for this repro.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 1000
COLS = 16
OUT_SHAPE = (ROWS, COLS)
OUT_STRIDE = (COLS, 1)
NUMEL = ROWS * COLS
CLASSIFICATION = "BANDWIDTH_BOUND"
XBLOCK = 256
NUM_WARPS = 4


from oracle_harness import (  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _lennard_jones_pointwise_kernel(
        arg0_ptr,
        arg2_ptr,
        arg1_ptr,
        out_ptr,
        arg0_stride0: tl.constexpr,
        arg2_stride0: tl.constexpr,
        arg1_stride0: tl.constexpr,
        X: tl.constexpr,
        N: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        mask = offsets < X
        rows = offsets // N
        cols = offsets % N

        arg2_vals = tl.load(arg2_ptr + rows * arg2_stride0, mask=mask, other=0.0)
        arg0_vals = tl.load(arg0_ptr + cols * arg0_stride0)
        arg1_vals = tl.load(arg1_ptr + cols * arg1_stride0)
        values = libdevice.tanh(arg2_vals * arg0_vals + arg1_vals)

        tl.store(out_ptr + offsets, values, mask=mask)


def oracle_forward(inputs):
    """Run the full Repro.forward scope with one Triton pointwise kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    arg0_1, arg2_1, arg1_1 = inputs
    for name, tensor, shape in (
        ("arg0_1", arg0_1, (COLS, 1)),
        ("arg2_1", arg2_1, (ROWS, 1)),
        ("arg1_1", arg1_1, (COLS,)),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor)!r}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape mismatch: got {tuple(tensor.shape)}, expected {shape}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} dtype mismatch: got {tensor.dtype}, expected torch.float32")
        if not tensor.is_cuda:
            raise ValueError("oracle_layout.py expects CUDA inputs")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(NUMEL, XBLOCK),)
    _lennard_jones_pointwise_kernel[grid](
        arg0_1,
        arg2_1,
        arg1_1,
        out,
        arg0_stride0=arg0_1.stride(0),
        arg2_stride0=arg2_1.stride(0),
        arg1_stride0=arg1_1.stride(0),
        X=NUMEL,
        N=COLS,
        XBLOCK=XBLOCK,
        num_warps=NUM_WARPS,
        num_stages=4,
    )
    return out


def _check_layout(output: torch.Tensor) -> bool:
    return tuple(output.shape) == OUT_SHAPE and output.stride() == OUT_STRIDE


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
        with torch.no_grad():
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout(layout_out)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_out.shape)} stride={layout_out.stride()})"
        )
        ok = ok and layout_ok
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
