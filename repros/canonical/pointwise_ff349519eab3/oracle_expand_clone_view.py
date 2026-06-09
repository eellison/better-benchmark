"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the expand-clone-view by loading each source element once and storing it into both duplicated positions of the final contiguous view, whereas Inductor lowers the clone of the zero-stride expanded tensor as a generic output-driven pointwise copy that reloads the same source value for each expanded output element; Inductor cannot do this today because its layout materialization path does not recognize expand-to-clone as a source-driven duplicate-store copy pattern; the fix is NEW_PATTERN: add a zero-stride expand clone lowering that emits one source load feeding multiple contiguous stores when the requested memory format requires materialization."""
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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=3),
        ],
        key=["N", "C"],
    )
    @triton.jit
    def _duplicate_expanded_dim_kernel(
        input_ptr,
        output_ptr,
        N: tl.constexpr,
        C: tl.constexpr,
        OUT_C: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0)

        rows = offsets // C
        cols = offsets - rows * C
        output_offsets = rows * OUT_C + cols
        tl.store(output_ptr + output_offsets, values, mask=mask)
        tl.store(output_ptr + output_offsets + C, values, mask=mask)


def _contiguous_3d_stride(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return (shape[1] * shape[2], shape[2], 1)


@oracle_impl(hardware="H100", shapes="(T([8, 4096, 256], f32), S([8, 4096, 2, 256]), S([8, 4096, 512]))")
def oracle_forward(inputs):
    """Run the exact Repro.forward computation for the captured inputs."""
    arg0_1, expand_shape, view_shape = inputs
    view_shape = tuple(int(dim) for dim in view_shape)
    expand_shape = tuple(int(dim) for dim in expand_shape)

    if arg0_1.device.type != "cuda" or triton is None:
        return (
            torch.ops.aten.unsqueeze.default(arg0_1, 2)
            .expand(expand_shape)
            .clone(memory_format=torch.contiguous_format)
            .view(view_shape)
        )

    if arg0_1.dtype != torch.float32:
        raise ValueError(f"expected float32 input, got {arg0_1.dtype}")
    if not arg0_1.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={arg0_1.stride()}")
    if len(expand_shape) != 4 or len(view_shape) != 3:
        raise ValueError(f"unexpected shape params: expand={expand_shape} view={view_shape}")
    if tuple(arg0_1.shape) != (expand_shape[0], expand_shape[1], expand_shape[3]):
        raise ValueError(
            f"input shape {tuple(arg0_1.shape)} is inconsistent with expand shape {expand_shape}"
        )
    if expand_shape[2] != 2 or view_shape != (expand_shape[0], expand_shape[1], 2 * expand_shape[3]):
        raise ValueError(f"unsupported expand/view shapes: expand={expand_shape} view={view_shape}")

    c = int(arg0_1.shape[2])
    out_c = int(view_shape[2])
    n = int(arg0_1.numel())
    output = torch.empty_strided(
        view_shape,
        _contiguous_3d_stride(view_shape),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)
    _duplicate_expanded_dim_kernel[grid](
        arg0_1,
        output,
        N=n,
        C=c,
        OUT_C=out_c,
    )
    return output


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
