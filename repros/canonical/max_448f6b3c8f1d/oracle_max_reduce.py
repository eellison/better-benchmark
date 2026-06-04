"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full aten.max.default over the canonical contiguous int64[1, 4096] input as a single one-program Triton block reduction that returns the same int64[] scalar as Repro.forward, whereas Inductor currently lowers this tiny whole-tensor max through its generic reduction template with launch-scale scheduler/codegen overhead around an otherwise single-kernel scalar reduction; Inductor cannot do this today because its reduction codegen does not select a specialized low-overhead integer scalar-reduction template for fixed small power-of-two tensors; the fix is NEW_PATTERN: add a small whole-tensor integer scalar-reduction lowering for max/min/sum-like reductions that emits one compact block reduction directly."""
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
N_ELEMENTS = 4096


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _max_i64_scalar_kernel(
        x_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        block_n: tl.constexpr,
    ):
        offsets = tl.arange(0, block_n)
        mask = offsets < n_elements
        values = tl.load(x_ptr + offsets, mask=mask, other=-9223372036854775808)
        max_value = tl.max(values, axis=0)
        tl.store(out_ptr, max_value)


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_max_reduce.py")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for oracle_max_reduce.py")


def oracle_forward(inputs, *, block_n: int = N_ELEMENTS) -> torch.Tensor:
    """Compute exactly Repro()(*make_inputs()) for the canonical scalar max."""
    _require_triton_cuda()
    if len(inputs) != 1:
        raise ValueError(f"expected one input tensor, got {len(inputs)} inputs")

    (arg0_1,) = inputs
    if not isinstance(arg0_1, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(arg0_1)!r}")
    if tuple(arg0_1.shape) != (1, N_ELEMENTS):
        raise ValueError(f"expected int64[1, {N_ELEMENTS}], got shape={tuple(arg0_1.shape)}")
    if arg0_1.dtype is not torch.int64:
        raise TypeError(f"expected torch.int64 input, got {arg0_1.dtype}")
    if not arg0_1.is_cuda:
        raise ValueError("oracle_max_reduce.py expects CUDA inputs")
    if block_n < arg0_1.numel():
        raise ValueError(f"block_n={block_n} is too small for {arg0_1.numel()} elements")
    if not arg0_1.is_contiguous():
        arg0_1 = arg0_1.contiguous()

    out = torch.empty((), device=arg0_1.device, dtype=torch.int64)
    _max_i64_scalar_kernel[(1,)](
        arg0_1,
        out,
        n_elements=arg0_1.numel(),
        block_n=block_n,
        num_warps=8,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument("--block-n", type=int, default=N_ELEMENTS, help="Triton reduction tile size")
    parser.add_argument("--no-skip-stochastic", action="store_true", help="Disable stochastic output skipping")
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    _require_triton_cuda()
    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            lambda values: oracle_forward(values, block_n=args.block_n),
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
        bench_fn = lambda values: oracle_forward(values, block_n=args.block_n)
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                bench_fn,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for {result['repro_id']} "
                        f"(ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = bench_oracle(
                bench_fn,
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
