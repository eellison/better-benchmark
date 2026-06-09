"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full Repro.forward scope by materializing the tangents_1 / 2 tensor once for the returned f32[1000,128] permute view and reducing those same scaled values once to produce both duplicate f32[1000] sum outputs, whereas Inductor lowers the two sibling sum(dim=0) consumers as separate reductions over the divided tensor alongside the layout-producing output; Inductor cannot do this today because its scheduler does not prove and merge duplicate reductions that share an identical producer while still preserving the separate output contract; the fix is ALGEBRAIC_ELIMINATION: canonicalize identical sibling reductions and views to one computed buffer and allow both returned sum outputs to intentionally alias that single result when aliasing is semantically unobservable."""
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

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _scale_and_sum_kernel(
        x_ptr,
        div_ptr,
        sum_ptr,
        n_cols: tl.constexpr,
        block_n: tl.constexpr,
        block_m: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_m = tl.arange(0, block_m)
        mask_n = offs_n < n_cols
        offsets = offs_m[:, None] * n_cols + offs_n[None, :]

        x = tl.load(x_ptr + offsets, mask=mask_n[None, :], other=0.0)
        scaled = x * 0.5
        tl.store(div_ptr + offsets, scaled, mask=mask_n[None, :])
        sums = tl.sum(scaled, axis=0)
        tl.store(sum_ptr + offs_n, sums, mask=mask_n)


@oracle_impl(hardware="H100", shapes="(T([128, 1000], f32), S([1000]), S([1000]))")
def oracle_forward(inputs):
    """Run the full oracle computation for Repro.forward."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    tangents_1 = inputs[0]
    if tangents_1.dim() != 2 or tangents_1.shape[0] != 128:
        raise ValueError(f"unexpected input shape for {REPRO_ID}: {tuple(tangents_1.shape)}")
    if tangents_1.dtype != torch.float32:
        raise ValueError(f"unexpected input dtype for {REPRO_ID}: {tangents_1.dtype}")
    if tangents_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")

    rows, cols = tangents_1.shape
    div_tensor = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=tangents_1.device,
        dtype=tangents_1.dtype,
    )
    sum_out = torch.empty((cols,), device=tangents_1.device, dtype=tangents_1.dtype)

    block_n = 16
    grid = (triton.cdiv(cols, block_n),)
    _scale_and_sum_kernel[grid](
        tangents_1,
        div_tensor,
        sum_out,
        cols,
        block_n,
        rows,
        num_warps=4,
    )
    # The two sum outputs are intentionally aliased: they are duplicate
    # reductions of the same producer with identical shape, dtype, and values.
    return (div_tensor.permute(1, 0), sum_out, sum_out)


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
