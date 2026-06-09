"""Gap diagnosis (classification: NEW_PATTERN): this oracle directly materializes the full view/permute/clone/view Visformer attention layout transform into the final contiguous output, whereas Inductor lowers the chain as a generic permuted clone materialization with rank-polymorphic layout-copy indexing; Inductor cannot do this today because its layout-copy codegen does not pattern-match this fixed head/channel/spatial reshape family and emit the direct affine transpose into the requested output view; the fix is NEW_PATTERN: add a Visformer-style layout materialization specialization that recognizes the view/permute/clone/view chain and writes the final contiguous output layout directly."""
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
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
    def _view_permute_clone_view_kernel(
        input_ptr,
        output_ptr,
        D2: tl.constexpr,
        D3: tl.constexpr,
        BLOCK_D2: tl.constexpr,
        BLOCK_D3: tl.constexpr,
    ):
        matrix = tl.program_id(0)
        d3_block = tl.program_id(1)
        d2_block = tl.program_id(2)

        d3 = d3_block * BLOCK_D3 + tl.arange(0, BLOCK_D3)
        d2 = d2_block * BLOCK_D2 + tl.arange(0, BLOCK_D2)

        input_base = matrix * D2 * D3
        output_base = matrix * D3 * D2

        input_offsets = input_base + d2[:, None] * D3 + d3[None, :]
        output_offsets = output_base + d3[:, None] * D2 + d2[None, :]
        input_mask = (d2[:, None] < D2) & (d3[None, :] < D3)
        output_mask = (d3[:, None] < D3) & (d2[None, :] < D2)

        values = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        tl.store(output_ptr + output_offsets, tl.trans(values), mask=output_mask)


def _as_shape(value) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_inputs(inputs):
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    x, shape0, shape1 = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"first input must be a tensor, got {type(x)!r}")
    if not x.is_cuda:
        raise ValueError("oracle requires a CUDA tensor input")
    if x.dtype is not torch.float32:
        raise TypeError(f"expected float32 input, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError(f"oracle expects the captured contiguous input layout, got stride={tuple(x.stride())}")

    view_shape = _as_shape(shape0)
    out_shape = _as_shape(shape1)
    if len(view_shape) != 4:
        raise ValueError(f"expected rank-4 first view shape, got {view_shape}")
    if _numel(tuple(x.shape)) != _numel(view_shape):
        raise ValueError(f"input shape {tuple(x.shape)} cannot view as {view_shape}")
    if _numel(out_shape) != _numel(view_shape):
        raise ValueError(f"output shape {out_shape} has different numel than {view_shape}")

    return x, view_shape, out_shape


def _block_sizes(d2: int, d3: int) -> tuple[int, int, int]:
    block_d2 = 32 if d2 <= 64 else 64
    block_d3 = 64 if d3 <= 128 else 128
    elements = block_d2 * block_d3
    num_warps = 8 if elements >= 8192 else 4
    return block_d2, block_d3, num_warps


@oracle_impl(hardware="H100", shapes="(T([768, 49, 128], f32), S([128, 6, 49, 128]), S([128, 768, 7, 7]))")
def oracle_forward(inputs):
    """Run the full view -> permute -> contiguous clone -> view scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, view_shape, out_shape = _validate_inputs(inputs)
    d0, d1, d2, d3 = view_shape
    output = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=x.dtype,
    )

    block_d2, block_d3, num_warps = _block_sizes(d2, d3)
    grid = (d0 * d1, triton.cdiv(d3, block_d3), triton.cdiv(d2, block_d2))
    _view_permute_clone_view_kernel[grid](
        x,
        output,
        D2=d2,
        D3=d3,
        BLOCK_D2=block_d2,
        BLOCK_D3=block_d3,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
