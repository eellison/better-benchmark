"""
Full-scope oracle for pointwise_8aed5a56d560.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the
complete Reformer shape-index result from `Repro.forward` by filling the
fresh `int64[4096]` iota buffer in one Triton launch and returning the same
`unsqueeze(0).expand(shape_param)` view, whereas Inductor already lowers this
tiny `prims.iota`/view region to equivalent launch-scale materialization plus
metadata; Inductor cannot materially do less today because the user-visible
CUDA int64 range output must exist and there is no producer/consumer chain,
scatter, reduction, or algebraic work to fuse away; the fix is
BANDWIDTH_BOUND: no scheduler/codegen change is indicated; parent disposition
should be driven by measured launch-floor status rather than a fusion or pattern
optimization claim.
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

IOTA_N = 4096
IOTA_SHAPE = (IOTA_N,)
IOTA_STRIDE = (1,)
IOTA_DTYPE = torch.int64
IOTA_DEVICE = torch.device("cuda", 0)
BLOCK_N = 1024


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _iota_i64_kernel(
        out_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        values = offsets.to(tl.int64)
        tl.store(out_ptr + offsets, values, mask=mask)


def _normalize_shape_param(shape_param: object) -> tuple[int, int]:
    if isinstance(shape_param, torch.Size):
        shape = tuple(shape_param)
    elif isinstance(shape_param, (list, tuple)):
        shape = tuple(int(dim) for dim in shape_param)
    else:
        raise TypeError(f"expected list/tuple/torch.Size shape input, got {type(shape_param)!r}")

    if len(shape) != 2 or shape[1] != IOTA_N:
        raise ValueError(f"expected expand shape (*, {IOTA_N}), got {shape}")
    return shape


def oracle_forward(inputs: list[object] | tuple[object, ...]) -> torch.Tensor:
    """Run the full iota/unsqueeze/expand repro scope."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one shape input, got {len(inputs)}")

    expand_shape = _normalize_shape_param(inputs[0])
    iota = torch.empty_strided(
        IOTA_SHAPE,
        IOTA_STRIDE,
        device=IOTA_DEVICE,
        dtype=IOTA_DTYPE,
    )
    _iota_i64_kernel[(triton.cdiv(IOTA_N, BLOCK_N),)](
        iota,
        N=IOTA_N,
        BLOCK_SIZE=BLOCK_N,
        num_warps=4,
        num_stages=1,
    )
    return iota.unsqueeze(0).expand(expand_shape)


def _check_layout(instance: torch.nn.Module, inputs: list[object] | tuple[object, ...]) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        oracle = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = (
        tuple(eager.shape) == tuple(oracle.shape)
        and eager.dtype == oracle.dtype
        and eager.stride() == oracle.stride()
        and eager.storage_offset() == oracle.storage_offset()
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(oracle.shape)} stride={oracle.stride()} "
        f"storage_offset={oracle.storage_offset()})"
    )
    return ok


def main() -> None:
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
        ok = ok and _check_layout(instance, inputs)
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
