"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete scaled attention-head layout scope by directly materializing the fresh contiguous `[B*H, S, D]` output from the contiguous `[B*S, H*D]` input with one Triton head-transpose copy that applies the scalar multiply, whereas Inductor already lowers the captured view/permute/mul/expand/clone/view chain to a comparable single materialization; Inductor cannot remove the dominant cost because the repro contract requires reading the full input and writing the fresh contiguous transposed output, so the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader layout-copy memory-traffic improvements move both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL_ROWS", "S", "H", "D", "BLOCK_D"],
    )
    @triton.jit
    def _scaled_head_layout_output_rows_kernel(
        input_ptr,
        output_ptr,
        TOTAL_ROWS: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_D)
        row_mask = rows < TOTAL_ROWS
        col_mask = cols < D

        batch_head = rows // S
        seq = rows - batch_head * S
        batch = batch_head // H
        head = batch_head - batch * H

        input_offsets = ((batch[:, None] * S + seq[:, None]) * H + head[:, None]) * D + cols[None, :]
        output_offsets = rows[:, None] * D + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values * SCALE, mask=mask)


SCALE = 0.3535533905932738


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_view_shape(value: Any, numel: int) -> tuple[int, ...]:
    dims = list(_shape_tuple(value))
    neg_one_count = dims.count(-1)
    if neg_one_count > 1:
        raise ValueError(f"only one inferred dimension is valid, got {dims}")
    if neg_one_count == 1:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"cannot infer shape {dims} for numel={numel}")
        dims[dims.index(-1)] = numel // known
    resolved = tuple(dims)
    product = 1
    for dim in resolved:
        product *= dim
    if product != numel:
        raise ValueError(f"shape {resolved} has {product} elements, expected {numel}")
    return resolved


def _resolve_expand_shape(value: Any, base_shape: tuple[int, ...]) -> tuple[int, ...]:
    dims = list(_shape_tuple(value))
    if len(dims) != len(base_shape):
        raise ValueError(f"expand shape {tuple(dims)} has rank {len(dims)}, expected {len(base_shape)}")
    resolved = tuple(base if dim == -1 else dim for dim, base in zip(dims, base_shape))
    for dim, base in zip(resolved, base_shape):
        if dim != base and base != 1:
            raise ValueError(f"expand shape {resolved} cannot expand base shape {base_shape}")
    return resolved


def _validate_and_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int], int, int, int, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    addmm_67, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(addmm_67, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not addmm_67.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if addmm_67.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {addmm_67.dtype}")
    if addmm_67.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects a rank-2 input, got shape={tuple(addmm_67.shape)}")
    if not addmm_67.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(addmm_67.stride())}")

    input_rows = int(addmm_67.shape[0])
    hidden = int(addmm_67.shape[1])
    numel = int(addmm_67.numel())

    batch, seq, shape_hidden = _resolve_view_shape(shape0, numel)
    if (batch * seq, shape_hidden) != (input_rows, hidden):
        raise ValueError(
            f"first view shape {(batch, seq, shape_hidden)} does not match input "
            f"shape={tuple(addmm_67.shape)}"
        )

    batch1, seq1, heads, head_dim = _resolve_view_shape(shape1, numel)
    if (batch1, seq1) != (batch, seq) or heads * head_dim != hidden:
        raise ValueError(
            f"head view shape {(batch1, seq1, heads, head_dim)} is incompatible "
            f"with first view {(batch, seq, hidden)}"
        )

    expand_shape = _resolve_expand_shape(shape2, (batch, heads, seq, head_dim))
    if expand_shape != (batch, heads, seq, head_dim):
        raise ValueError(f"unexpected expand shape {expand_shape}")

    output_shape = _resolve_view_shape(shape3, numel)
    expected_output_shape = (batch * heads, seq, head_dim)
    if output_shape != expected_output_shape:
        raise ValueError(f"unexpected output shape {output_shape}, expected {expected_output_shape}")

    output_stride = (seq * head_dim, head_dim, 1)
    total_rows = batch * heads * seq
    return addmm_67, output_shape, output_stride, total_rows, seq, heads, head_dim


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
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    addmm_67, output_shape, output_stride, total_rows, seq, heads, head_dim = _validate_and_layout(inputs)
    output = torch.empty_strided(
        output_shape,
        output_stride,
        device=addmm_67.device,
        dtype=addmm_67.dtype,
    )

    block_d = triton.next_power_of_2(head_dim)
    grid = lambda meta: (triton.cdiv(total_rows, meta["BLOCK_ROWS"]),)
    _scaled_head_layout_output_rows_kernel[grid](
        addmm_67,
        output,
        TOTAL_ROWS=total_rows,
        S=seq,
        H=heads,
        D=head_dim,
        SCALE=SCALE,
        BLOCK_D=block_d,
    )
    return output


def _check_layout(reference: torch.Tensor, output: torch.Tensor) -> bool:
    return (
        tuple(output.shape) == tuple(reference.shape)
        and tuple(output.stride()) == tuple(reference.stride())
        and output.dtype is reference.dtype
        and output.device == reference.device
        and output.storage_offset() == reference.storage_offset()
    )


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
        with torch.no_grad():
            reference = instance(*inputs)
            layout_output = oracle_forward(inputs)
            if layout_output.is_cuda:
                torch.cuda.synchronize()
        layout_ok = _check_layout(reference, layout_output)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_output.shape)} stride={list(layout_output.stride())} "
            f"dtype={layout_output.dtype})"
        )
        ok = ok and layout_ok
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
