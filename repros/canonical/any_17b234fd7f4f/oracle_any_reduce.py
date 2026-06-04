"""
Full-scope Triton oracle for any_17b234fd7f4f.

Pattern:
  gt(arg0, 0) -> view([8192]) -> any() -> bool scalar

Classification: BANDWIDTH_BOUND.  The full computation is a one-pass 8192
element load plus scalar boolean reduction; Inductor already emits a single
small reduction kernel for this shape, so there is no scheduler-fusion issue to
close beyond launch/reduction overhead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch._inductor.runtime import triton_helpers

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
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs() -> tuple[object, ...]:
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _gt_zero_any_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)[None, :]
        mask = offsets < n_elements
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        positive = (values > 0.0) & mask
        any_positive = triton_helpers.any(positive.to(tl.int8), 1)[:, None]
        tl.store(out_ptr + tl.full((1, 1), 0, tl.int32), any_positive)


def _shape_numel(shape: object) -> int:
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"{REPRO_ID} expected view shape as list/tuple, got {type(shape).__name__}")
    numel = 1
    for dim in shape:
        if not isinstance(dim, int):
            raise ValueError(f"{REPRO_ID} expected integer view dims, got {shape!r}")
        numel *= dim
    return numel


def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Run gt(arg0, 0), view, and any as one Triton reduction kernel."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, view_shape = inputs
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{REPRO_ID} expected tensor input 0, got {type(x).__name__}")
    if not x.is_cuda:
        raise RuntimeError("CUDA input is required for this Triton oracle")
    if not x.is_contiguous():
        x = x.contiguous()

    n_elements = x.numel()
    if _shape_numel(view_shape) != n_elements:
        raise ValueError(
            f"{REPRO_ID} view shape {view_shape!r} does not match input numel {n_elements}"
        )
    if n_elements < 1:
        raise ValueError(f"{REPRO_ID} expects a non-empty reduction")

    block = triton.next_power_of_2(n_elements)
    out = torch.empty((), device=x.device, dtype=torch.bool)
    _gt_zero_any_kernel[(1,)](x, out, n_elements, BLOCK=block, num_warps=8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Oracle for {REPRO_ID}")
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
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
        print(f"Benchmarking {REPRO_ID}... classification={CLASSIFICATION}")
        if args.all_shapes:
            bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
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
