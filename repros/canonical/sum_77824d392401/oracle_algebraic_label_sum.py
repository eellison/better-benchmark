"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete dense cross-entropy backward output while deriving the row reduction of the expanded one-hot target tensor directly from the label and scalar gradient, whereas Inductor emits a generic row reduction that scans the full 262144-column equality expansion before the dense exponential epilogue; Inductor cannot do this today because it does not recognize this one-hot sum as a closed-form indexed scalar feeding the same full-row output; the fix is ALGEBRAIC_ELIMINATION: add an FX or scheduler rewrite for one-hot equality sums so the generated kernel keeps only the dense epilogue with the original row labels, f32 subtraction order, bf16 producer rounding, libdevice exponential, and bf16 output store."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None
    libdevice = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_BLOCK_M = 1
DEFAULT_BLOCK_N = 2048


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None and libdevice is not None:

    @triton.jit
    def _xent_backward_epilogue_kernel(
        grad_scalar_ptr,
        labels_ptr,
        logits_ptr,
        shift0_ptr,
        shift1_ptr,
        out_ptr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        col_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        mask = col_offsets < N
        offsets = row_offsets * N + col_offsets

        labels = tl.load(labels_ptr + row_offsets)
        scale = tl.load(grad_scalar_ptr).to(tl.float32)
        valid = (labels != -100) & (labels >= 0) & (labels < N)
        row_sum = tl.where(valid, -scale, 0.0)
        sparse = tl.where((col_offsets == labels) & valid, -scale, 0.0)

        logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        shift0 = tl.load(shift0_ptr + row_offsets).to(tl.float32)
        shift1 = tl.load(shift1_ptr + row_offsets).to(tl.float32)
        centered = logits - shift0
        centered = centered - shift1
        rounded = centered.to(tl.bfloat16).to(tl.float32)
        dense = libdevice.exp(rounded) * row_sum
        out = sparse - dense
        tl.store(out_ptr + offsets, out, mask=mask)


def _validate_inputs(inputs):
    if triton is None or libdevice is None:
        raise RuntimeError("Triton and Inductor libdevice are required for this oracle")

    (
        grad_scalar,
        labels,
        logits,
        shift0,
        shift1,
        shape_m,
        shape_view,
        shape_out,
    ) = inputs

    if not isinstance(shape_m, (list, tuple)) or not isinstance(shape_view, (list, tuple)):
        raise ValueError("unexpected shape parameter type")
    if not isinstance(shape_out, (list, tuple)):
        raise ValueError("unexpected output shape parameter type")

    m, n = int(shape_out[0]), int(shape_out[1])
    if shape_m != [m] or shape_view != [1, n]:
        raise ValueError("shape parameters do not match the captured output shape")
    if grad_scalar.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if grad_scalar.shape != () or grad_scalar.dtype != torch.bfloat16:
        raise ValueError("expected bf16 scalar gradient input")
    if tuple(labels.shape) != (m,) or labels.dtype != torch.int64 or labels.stride() != (1,):
        raise ValueError("expected contiguous i64 labels")
    if tuple(logits.shape) != (m, n) or logits.dtype != torch.bfloat16:
        raise ValueError("expected bf16 logits with captured shape")
    if logits.stride() != (n, 1):
        raise ValueError("expected contiguous logits")
    if tuple(shift0.shape) != (m, 1) or shift0.dtype != torch.float32 or shift0.stride() != (1, 1):
        raise ValueError("expected captured f32 shift0 layout")
    if tuple(shift1.shape) != (m, 1) or shift1.dtype != torch.float32 or shift1.stride() != (1, 1):
        raise ValueError("expected captured f32 shift1 layout")
    return m, n, grad_scalar, labels, logits, shift0, shift1


@oracle_impl(hardware="H100", shapes="(T([], bf16), T([8192], i64), T([8192, 262144], bf16), T([8192, 1], f32), T([8192, 1], f32), S([8192]), S([1, 262144]), S([8192, 262144]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope and return the dense bf16 output."""
    m, n, grad_scalar, labels, logits, shift0, shift1 = _validate_inputs(inputs)
    out = torch.empty_strided((m, n), (n, 1), device=logits.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, DEFAULT_BLOCK_M), triton.cdiv(n, DEFAULT_BLOCK_N))
    _xent_backward_epilogue_kernel[grid](
        grad_scalar,
        labels,
        logits,
        shift0,
        shift1,
        out,
        N=n,
        BLOCK_M=DEFAULT_BLOCK_M,
        BLOCK_N=DEFAULT_BLOCK_N,
        num_warps=4,
    )
    return out


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
