"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DeBERTa attention-head layout and scalar-divide scope by directly materializing the final contiguous `[B*H, S, D]` output from the contiguous `[B*S, H*D]` projection with one Triton transpose/divide kernel, whereas Inductor already has to perform the same full input read and fresh output write for the captured view/permute/clone/div/permute chain; Inductor cannot remove the dominant cost because the repro contract requires the layout materialization and scalar division over every element, so the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader layout-copy memory-traffic or launch-overhead improvements move both implementations."""
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
    def _scaled_head_layout_div_kernel(
        input_ptr,
        scale_ptr,
        output_ptr,
        TOTAL_ROWS: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
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
        inv_scale = 1.0 / tl.load(scale_ptr)
        tl.store(output_ptr + output_offsets, values * inv_scale, mask=mask)


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


def _validate_and_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], tuple[int, int, int], int, int, int, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    addmm_139, full, shape0, shape1, shape2 = inputs
    if not isinstance(addmm_139, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor")
    if not isinstance(full, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 1 must be a tensor")
    if not addmm_139.is_cuda or not full.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA tensor inputs")
    if addmm_139.dtype != torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input 0, got {addmm_139.dtype}")
    if full.dtype != torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 scalar, got {full.dtype}")
    if addmm_139.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects a rank-2 input, got shape={tuple(addmm_139.shape)}")
    if full.ndim != 0:
        raise ValueError(f"{REPRO_ID} expects a rank-0 scalar divisor, got shape={tuple(full.shape)}")
    if not addmm_139.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input 0, got stride={tuple(addmm_139.stride())}")

    input_rows = int(addmm_139.shape[0])
    hidden = int(addmm_139.shape[1])
    numel = int(addmm_139.numel())

    batch, seq, shape_hidden = _resolve_view_shape(shape0, numel)
    if (batch * seq, shape_hidden) != (input_rows, hidden):
        raise ValueError(
            f"first view shape {(batch, seq, shape_hidden)} does not match input "
            f"shape={tuple(addmm_139.shape)}"
        )

    batch1, seq1, heads, head_dim = _resolve_view_shape(shape1, numel)
    if (batch1, seq1) != (batch, seq) or heads * head_dim != hidden:
        raise ValueError(
            f"head view shape {(batch1, seq1, heads, head_dim)} is incompatible "
            f"with first view {(batch, seq, hidden)}"
        )

    output_shape = _resolve_view_shape(shape2, numel)
    expected_output_shape = (batch * heads, seq, head_dim)
    if output_shape != expected_output_shape:
        raise ValueError(f"unexpected output shape {output_shape}, expected {expected_output_shape}")

    output_stride = (seq * head_dim, head_dim, 1)
    total_rows = batch * heads * seq
    return addmm_139, full, output_shape, output_stride, total_rows, seq, heads, head_dim


@oracle_impl(hardware="H100", shapes="(T([4096, 1536], f32), T([], f32), S([8, 512, 1536]), S([8, 512, 24, -1]), S([-1, 512, 64]))")
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

    addmm_139, full, output_shape, output_stride, total_rows, seq, heads, head_dim = _validate_and_layout(inputs)
    output = torch.empty_strided(
        output_shape,
        output_stride,
        device=addmm_139.device,
        dtype=addmm_139.dtype,
    )

    block_d = triton.next_power_of_2(head_dim)
    grid = lambda meta: (triton.cdiv(total_rows, meta["BLOCK_ROWS"]),)
    _scaled_head_layout_div_kernel[grid](
        addmm_139,
        full,
        output,
        TOTAL_ROWS=total_rows,
        S=seq,
        H=heads,
        D=head_dim,
        BLOCK_D=block_d,
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
