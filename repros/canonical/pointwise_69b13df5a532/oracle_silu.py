"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete 3072-element fp16 SiLU repro in one tiny Triton pointwise kernel, including the fp32 cast, neg/exp/add/div expression, fp16 output cast, and captured contiguous [64, 48, 1, 1] layout, whereas Inductor currently compiles the same decomposed prims/aten graph as a generic fused pointwise kernel at the launch floor; Inductor cannot do this today because its pattern/codegen stack has no dedicated tiny-contiguous SiLU lowering that bypasses generic pointwise scheduling overhead; the fix is NEW_PATTERN: add a guarded SiLU pointwise template for small contiguous tensors or make the generic pointwise schedule select the same minimal one-block kernel."""
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
REPRO_PATH = REPRO_DIR / "repro.py"

INPUT_SHAPE = (64, 48, 1, 1)
INPUT_DTYPE = torch.float16
NUMEL = 64 * 48
BLOCK = 4096
CLASSIFICATION = "NEW_PATTERN"


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _silu_f16_kernel(
        x_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = x / (tl.exp(-x) + 1.0)
        tl.store(out_ptr + offsets, y, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([64, 48, 1, 1], f16))")
def oracle_forward(inputs: list[object] | tuple[object, ...]) -> torch.Tensor:
    """Run the full repro computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_silu.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (x,) = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(x)!r}")
    if tuple(x.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if x.dtype is not INPUT_DTYPE:
        raise ValueError(f"unexpected input dtype: {x.dtype}")
    if not x.is_cuda:
        raise ValueError("oracle_silu.py expects CUDA inputs")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={x.stride()}")

    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float16,
    )
    _silu_f16_kernel[(triton.cdiv(NUMEL, BLOCK),)](
        x,
        out,
        n_elements=NUMEL,
        BLOCK_SIZE=BLOCK,
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
        with torch.no_grad():
            out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = tuple(out.shape) == INPUT_SHAPE and out.stride() == inputs[0].stride()
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(out.shape)} stride={out.stride()})"
        )
        ok = ok and layout_ok
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
