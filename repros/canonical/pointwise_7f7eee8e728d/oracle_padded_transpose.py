"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete padded transpose as one tiled layout-transform kernel that writes the final contiguous `[K, 30524]` result including the two zero pad columns, whereas Inductor's compiled full-scope copy lands at essentially the same CUDAGraph timing; Inductor cannot materially improve this repro through additional fusion because the graph has no arithmetic reuse and is dominated by reading and writing the full tensor; the fix is BANDWIDTH_BOUND: record this repro as at floor rather than pursue a scheduler/codegen change."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

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


REPRO_ID = "pointwise_7f7eee8e728d"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BLOCK_M = 64
BLOCK_N = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _padded_transpose_kernel(
        in_ptr,
        out_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        OUT_M: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        offs_k = pid_k * BLOCK_N_ + tl.arange(0, BLOCK_N_)

        load_offsets = offs_m[:, None] * K + offs_k[None, :]
        load_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        values = tl.load(in_ptr + load_offsets, mask=load_mask, other=0.0)

        store_offsets = offs_k[:, None] * OUT_M + offs_m[None, :]
        store_mask = (offs_k[:, None] < K) & (offs_m[None, :] < OUT_M)
        tl.store(out_ptr + store_offsets, tl.trans(values), mask=store_mask)


def _validate_input(x: torch.Tensor) -> tuple[int, int]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(x)!r}")
    if x.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if x.ndim != 2:
        raise ValueError(f"expected a rank-2 input, got shape={tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise ValueError(f"expected float32 input, got dtype={x.dtype}")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={x.stride()}")
    return int(x.shape[0]), int(x.shape[1])


@oracle_impl(hardware="H100", shapes="(T([30522, 768], f32))")
def oracle_forward(inputs):
    """Compute the exact Repro.forward output for the full captured input tuple."""
    if len(inputs) != 1:
        raise ValueError(f"expected one input, got {len(inputs)}")
    x = inputs[0]
    m, k = _validate_input(x)
    out_m = m + 2
    out = torch.empty_strided((k, out_m), (out_m, 1), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(out_m, BLOCK_M), triton.cdiv(k, BLOCK_N))
    _padded_transpose_kernel[grid](
        x,
        out,
        M=m,
        K=k,
        OUT_M=out_m,
        BLOCK_M_=BLOCK_M,
        BLOCK_N_=BLOCK_N,
        num_warps=8,
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify correctness against eager Repro",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark oracle vs torch.compile",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for correctness check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for correctness check",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup iterations for benchmark",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=200,
        help="Repetitions for benchmark",
    )
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Benchmark across all shapes from shapes.txt",
    )
    parser.add_argument(
        "--show-hw",
        action="store_true",
        help="Print GPU hardware info and exit",
    )
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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        "WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
