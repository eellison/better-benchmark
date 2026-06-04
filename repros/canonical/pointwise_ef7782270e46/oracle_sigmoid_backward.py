"""
Oracle for pointwise_ef7782270e46.

Gap diagnosis: Classification: BANDWIDTH_BOUND. This oracle computes the full
`arg23 * arg19 * (1 - arg19)` sigmoid-backward pointwise expression over all
1024 f32 elements in one Triton kernel and writes the fresh output directly,
while Inductor already lowers this captured graph to the same one-launch
pointwise work; the measured gap is therefore residual launch, allocation, and
memory traffic overhead, with no scheduler fusion opportunity inside this
single-output expression.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
    triton = None
    tl = None

from oracle_harness import (
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

N_ELEMENTS = 1024
BLOCK_SIZE = 1024


def get_inputs() -> tuple[object, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _sigmoid_backward_kernel(
        arg19_ptr,
        arg23_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        arg19 = tl.load(arg19_ptr + offsets, mask=mask, other=0.0)
        arg23 = tl.load(arg23_ptr + offsets, mask=mask, other=0.0)
        out = arg23 * arg19 * (1.0 - arg19)
        tl.store(out_ptr + offsets, out, mask=mask)


def _validate_inputs(arg19: torch.Tensor, arg23: torch.Tensor) -> None:
    expected_shape = (1024, 1, 1, 1)
    if tuple(arg19.shape) != expected_shape or tuple(arg23.shape) != expected_shape:
        raise ValueError(
            f"expected both inputs to have shape {expected_shape}, "
            f"got {tuple(arg19.shape)} and {tuple(arg23.shape)}"
        )
    if arg19.dtype is not torch.float32 or arg23.dtype is not torch.float32:
        raise TypeError(f"expected float32 tensors, got {arg19.dtype} and {arg23.dtype}")
    if not arg19.is_cuda or not arg23.is_cuda:
        raise ValueError("oracle requires CUDA tensors")
    if not arg19.is_contiguous() or not arg23.is_contiguous():
        raise ValueError("oracle expects contiguous tensors from the repro shape config")


def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope with a single Triton pointwise kernel."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    arg19, arg23 = inputs
    if not isinstance(arg19, torch.Tensor) or not isinstance(arg23, torch.Tensor):
        raise TypeError("expected both inputs to be tensors")
    _validate_inputs(arg19, arg23)

    out = torch.empty_strided(
        (1024, 1, 1, 1),
        (1, 1, 1, 1),
        dtype=arg19.dtype,
        device=arg19.device,
    )
    _sigmoid_backward_kernel[(triton.cdiv(N_ELEMENTS, BLOCK_SIZE),)](
        arg19,
        arg23,
        out,
        N_ELEMENTS,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
                        f"WARNING: oracle is slower than compile for "
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
