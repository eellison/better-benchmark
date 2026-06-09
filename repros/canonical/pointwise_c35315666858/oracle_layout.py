"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full zero-input Repro.forward by folding `full([16, 512], 1.0)` through both unsqueeze views and `1.0 - tensor` into one fresh contiguous `float32[16, 1, 1, 512]` positive-zero output filled by a single Triton kernel, whereas Inductor currently lowers the graph as a generic generated pointwise constant-sub/fill path for the final tensor instead of canonicalizing the view-threaded scalar-minus-full expression to a zero tensor; Inductor cannot do this today because its algebraic simplifier does not propagate tensor-valued constants through view-only unsqueeze nodes before pointwise scheduling/codegen; the fix is ALGEBRAIC_ELIMINATION: add an Inductor rewrite that propagates constant full tensors through view ops and replaces `scalar - full(c)` with a direct `full(scalar - c)` of the final layout."""
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

OUT_SHAPE = (16, 1, 1, 512)
OUT_STRIDE = (512, 512, 512, 1)
OUT_NUMEL = OUT_SHAPE[0] * OUT_SHAPE[1] * OUT_SHAPE[2] * OUT_SHAPE[3]
BLOCK = 512
GRID = (OUT_NUMEL // BLOCK,)
OUT_DEVICE = torch.device("cuda", 0)


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _zero_fill_f32_kernel(
        out_ptr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        tl.store(out_ptr + offsets, 0.0)

    _zero_fill_f32_launcher = _zero_fill_f32_kernel[GRID]
else:

    def _zero_fill_f32_launcher(*_args, **_kwargs):
        raise RuntimeError("Triton is required for the timed oracle")


@oracle_impl(hardware="H100", shapes="()")
def oracle_forward(inputs: list[object] | tuple[object, ...]) -> torch.Tensor:
    """Run the full repro computation and return the fresh positive-zero tensor."""
    if inputs:
        raise AssertionError(f"expected no inputs, got {len(inputs)}")
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    out = torch.empty(
        OUT_SHAPE,
        device=OUT_DEVICE,
        dtype=torch.float32,
    )
    _zero_fill_f32_launcher(
        out,
        block_size=BLOCK,
        num_warps=1,
        num_stages=1,
    )
    return out


def _check_layout_and_bits(
    instance: torch.nn.Module,
    inputs: list[object] | tuple[object, ...],
) -> bool:
    with torch.no_grad():
        eager_out = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    layout_ok = (
        tuple(oracle_out.shape) == OUT_SHAPE
        and oracle_out.stride() == OUT_STRIDE
        and oracle_out.dtype == torch.float32
        and oracle_out.storage_offset() == 0
    )
    bits_ok = torch.equal(eager_out.view(torch.uint32), oracle_out.view(torch.uint32))
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} stride={oracle_out.stride()} "
        f"dtype={oracle_out.dtype})"
    )
    print(f"  output 0 bit_pattern: {'PASS' if bits_ok else 'FAIL'}")
    return layout_ok and bits_ok


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
        ok = _check_layout_and_bits(instance, inputs) and ok
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
