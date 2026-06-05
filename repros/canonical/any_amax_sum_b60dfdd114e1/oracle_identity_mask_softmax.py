"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Albert masked softmax returned by Repro.forward, including the [512,512,512] to [8,64,512,512] view, elimination of the generated iota >= 0 all-true additive mask, stable last-dimension softmax, all-minus-inf row fallback to zeros, and final contiguous [512,512,512] view in one Triton row kernel, whereas Inductor already benchmarks within the 5% floor of this full materialized softmax despite lowering the decomposed iota/where/add/eq/any/amax/sub/exp/sum/div/where graph; Inductor cannot expose a meaningful additional win here because the required returned output forces reading and writing the full 512-wide softmax tensor and the remaining work is dominated by the row-softmax computation itself; the fix is BANDWIDTH_BOUND: do not treat this as an actionable priority gap unless a consumer can be fused to avoid materializing the full softmax output."""
from __future__ import annotations

import argparse
import math
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


LOG2E = 1.4426950408889634


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _resolve_view_shape(shape_param, numel: int) -> tuple[int, ...]:
    dims = [int(dim) for dim in shape_param]
    inferred = None
    known = 1
    for idx, dim in enumerate(dims):
        if dim == -1:
            if inferred is not None:
                raise ValueError(f"multiple inferred dimensions in shape {dims}")
            inferred = idx
        else:
            known *= dim

    if inferred is None:
        if known != numel:
            raise ValueError(f"shape {dims} has {known} elements, expected {numel}")
    else:
        if known == 0 or numel % known != 0:
            raise ValueError(f"shape {dims} cannot view {numel} elements")
        dims[inferred] = numel // known
    return tuple(dims)


def _resolve_expand_shape(shape_param, source_shape: tuple[int, ...]) -> tuple[int, ...]:
    dims = [int(dim) for dim in shape_param]
    if len(dims) != len(source_shape):
        raise ValueError(f"expand rank {len(dims)} does not match source rank {len(source_shape)}")
    return tuple(source if dim == -1 else dim for dim, source in zip(dims, source_shape))


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_scope(
    bmm: torch.Tensor,
    shape_param_0,
    shape_param_1,
    shape_param_2,
    shape_param_3,
) -> tuple[tuple[int, ...], int, int]:
    if not isinstance(bmm, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(bmm).__name__}")
    if bmm.dtype != torch.float32:
        raise TypeError(f"expected fp32 input, got {bmm.dtype}")
    if not bmm.is_contiguous():
        raise ValueError("expected contiguous input matching repro make_inputs")

    numel = bmm.numel()
    view_shape = _resolve_view_shape(shape_param_0, numel)
    mask_shape = _resolve_expand_shape(shape_param_1, (1, 1, 512, 1))
    expand_shape = _resolve_expand_shape(shape_param_2, view_shape)
    output_shape = _resolve_view_shape(shape_param_3, numel)

    if len(view_shape) != 4:
        raise ValueError(f"expected rank-4 softmax view shape, got {view_shape}")
    if mask_shape != (view_shape[0], 1, view_shape[2], view_shape[3]):
        raise ValueError(f"mask expand shape {mask_shape} is not the repro's all-true mask")
    if expand_shape != view_shape:
        raise ValueError(f"final expand shape {expand_shape} does not match softmax shape {view_shape}")
    if math.prod(output_shape) != numel:
        raise ValueError(f"output shape {output_shape} does not preserve input numel {numel}")

    n_cols = view_shape[-1]
    if n_cols <= 0:
        raise ValueError(f"invalid softmax column count {n_cols}")
    n_rows = numel // n_cols
    return output_shape, n_rows, n_cols


def _aten_fallback(
    bmm: torch.Tensor,
    shape_param_0,
    shape_param_2,
    shape_param_3,
) -> torch.Tensor:
    view = bmm.view(_shape_tuple(shape_param_0))
    valid = (view != -float("inf")).any(dim=-1, keepdim=True)
    softmax = torch.softmax(view, dim=-1)
    guarded = torch.where(valid, softmax, torch.zeros_like(softmax))
    expanded = guarded.expand(_shape_tuple(shape_param_2))
    return expanded.view(_shape_tuple(shape_param_3))


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=8, num_stages=4),
        ],
        key=["n_rows", "n_cols"],
    )
    @triton.jit
    def oracle_kernel(
        input_ptr,
        output_ptr,
        n_rows: tl.constexpr,
        n_cols: tl.constexpr,
        block_cols: tl.constexpr,
        log2e: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, block_cols)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
        offsets = rows[:, None] * n_cols + cols[None, :]

        vals = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(vals, axis=1)
        row_all_neg_inf = row_max == -float("inf")

        shifted = vals - row_max[:, None]
        shifted = tl.where(mask, shifted, -float("inf"))
        shifted = tl.where(row_all_neg_inf[:, None] & mask, 0.0, shifted)
        numer = tl.exp2(shifted * log2e)
        denom = tl.sum(numer, axis=1)
        out = numer / denom[:, None]
        out = tl.where(row_all_neg_inf[:, None], 0.0, out)

        tl.store(output_ptr + offsets, out, mask=mask)


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    bmm, shape_param_0, shape_param_1, shape_param_2, shape_param_3 = inputs
    output_shape, n_rows, n_cols = _validate_scope(
        bmm,
        shape_param_0,
        shape_param_1,
        shape_param_2,
        shape_param_3,
    )

    if triton is None or not bmm.is_cuda:
        return _aten_fallback(bmm, shape_param_0, shape_param_2, shape_param_3)

    output = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=bmm.device,
        dtype=bmm.dtype,
    )
    block_cols = triton.next_power_of_2(n_cols)
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_ROWS"]),)
    oracle_kernel[grid](
        bmm,
        output,
        n_rows=n_rows,
        n_cols=n_cols,
        block_cols=block_cols,
        log2e=LOG2E,
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
