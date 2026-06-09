"""Oracle for pointwise_d76b39ac2742.

Gap diagnosis (classification: BANDWIDTH_BOUND): this full-scope at-floor oracle folds the `prims.iota(512)`, `[1, 512]` expand, `+ 2`, and embedding lookup into one Triton affine copy from rows `2..513` of the fp16 table into the required contiguous `[1, 512, 768]` output, whereas Inductor lowers the decomposed iota/add/embedding pattern to essentially the same materializing gather/copy work and already matches this floor; Inductor cannot eliminate the remaining cost today because `aten.embedding` semantically returns a fresh dense tensor and this repro has no consumer into which the table slice can be fused, so the required change is BANDWIDTH_BOUND: recognize affine embedding indices as a contiguous-slice copy where possible, but expect only launch/allocation and memory-traffic-level gains unless a downstream consumer enables fusion.
"""
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


REPRO_ID = "pointwise_d76b39ac2742"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

SEQ_LEN = 512
HIDDEN = 768
ROW_OFFSET = 2
TOTAL = SEQ_LEN * HIDDEN


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        ],
        key=[],
    )
    @triton.jit
    def _embedding_iota_copy_kernel(
        table_ptr,
        out_ptr,
        total: tl.constexpr,
        hidden: tl.constexpr,
        row_offset: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        rows = offsets // hidden
        cols = offsets - rows * hidden
        values = tl.load(table_ptr + (rows + row_offset) * hidden + cols, mask=mask)
        tl.store(out_ptr + offsets, values, mask=mask)


def _validate_input(arg0_1: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if arg0_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA input")
    if arg0_1.dtype != torch.float16:
        raise TypeError(f"expected fp16 embedding table, got {arg0_1.dtype}")
    if tuple(arg0_1.shape) != (1026, HIDDEN):
        raise ValueError(f"unexpected embedding table shape: {tuple(arg0_1.shape)}")
    if not arg0_1.is_contiguous():
        raise ValueError("oracle expects the captured embedding table to be contiguous")


def oracle_embedding(arg0_1: torch.Tensor) -> torch.Tensor:
    """Compute the complete Repro.forward embedding result with a Triton kernel."""
    _validate_input(arg0_1)
    out = torch.empty_strided(
        (1, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = lambda meta: (triton.cdiv(TOTAL, meta["BLOCK_SIZE"]),)
    _embedding_iota_copy_kernel[grid](arg0_1, out, TOTAL, HIDDEN, ROW_OFFSET)
    return out


@oracle_impl(hardware="H100", shapes="(T([1026, 768], f16))")
def oracle_forward(inputs):
    """Run the full-scope oracle for Repro.forward."""
    (arg0_1,) = inputs
    return oracle_embedding(arg0_1)


def main():
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
