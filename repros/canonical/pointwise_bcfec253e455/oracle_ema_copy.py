"""
Oracle for pointwise_bcfec253e455.

Gap diagnosis: Classification: BANDWIDTH_BOUND. This oracle computes the full
`arg321 = arg321 * 0.999 + arg322 * 0.001` update in one Triton pointwise
kernel, stores directly into `arg321`, and returns that mutated tensor exactly
like `aten.copy_`, while Inductor already lowers the captured graph to the same
one-launch in-place pointwise shape with only generic launch/scheduling overhead
left; Inductor cannot materially improve this today because the visible work is
one read of each input plus one write of the mutated input, so the fix
classification is BANDWIDTH_BOUND rather than a missing fusion or new template.
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
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"


def get_inputs() -> tuple[object, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _ema_copy_kernel(
        arg321_ptr,
        arg322_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        arg321 = tl.load(arg321_ptr + offsets, mask=mask, other=0.0)
        arg322 = tl.load(arg322_ptr + offsets, mask=mask, other=0.0)
        updated = arg321 * 0.999 + arg322 * 0.0010000000000000009
        tl.store(arg321_ptr + offsets, updated, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([128], f32), T([128], f32))")
def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Run the full pointwise EMA copy_ scope and return the mutated input."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    arg321, arg322 = inputs
    if not isinstance(arg321, torch.Tensor) or not isinstance(arg322, torch.Tensor):
        raise TypeError("expected both inputs to be tensors")
    if arg321.shape != arg322.shape:
        raise ValueError(f"shape mismatch: {tuple(arg321.shape)} vs {tuple(arg322.shape)}")
    if arg321.dtype is not torch.float32 or arg322.dtype is not torch.float32:
        raise TypeError(f"expected float32 tensors, got {arg321.dtype} and {arg322.dtype}")
    if not arg321.is_cuda or not arg322.is_cuda:
        raise ValueError("oracle requires CUDA tensors")
    if not arg321.is_contiguous() or not arg322.is_contiguous():
        raise ValueError("oracle expects contiguous tensors from the repro shape config")

    n_elements = arg321.numel()
    if n_elements == 0:
        return arg321

    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _ema_copy_kernel[grid](
        arg321,
        arg322,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return arg321


def main() -> None:
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
