"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Longformer `permute.clone.view` layout scope by directly materializing the fresh contiguous `[1024, B, 768]` clone buffer with a sequence-major Triton copy and returning the required `[1024 * B, 768]` view, whereas Inductor lowers the same mandatory layout materialization through its generic pointwise copy schedule; Inductor cannot remove the dominant work today because the repro contract requires a non-aliasing contiguous clone with one full read and one full write of the input tensor; the fix is BANDWIDTH_BOUND: record this as at floor unless broader layout-copy indexing, launch, or memory-bandwidth improvements move both implementations together."""
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
            triton.Config({"BLOCK_ROWS": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL_ROWS", "B"],
    )
    @triton.jit
    def _seq_batch_layout_kernel(
        input_ptr,
        clone_ptr,
        TOTAL_ROWS: tl.constexpr,
        B: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < TOTAL_ROWS
        seq = rows // B
        batch = rows - seq * B

        cols0 = tl.arange(0, 512)
        input_offsets0 = batch[:, None] * S * H + seq[:, None] * H + cols0[None, :]
        clone_offsets0 = rows[:, None] * H + cols0[None, :]
        values0 = tl.load(input_ptr + input_offsets0, mask=row_mask[:, None])
        tl.store(clone_ptr + clone_offsets0, values0, mask=row_mask[:, None])

        cols1 = 512 + tl.arange(0, 256)
        input_offsets1 = batch[:, None] * S * H + seq[:, None] * H + cols1[None, :]
        clone_offsets1 = rows[:, None] * H + cols1[None, :]
        values1 = tl.load(input_ptr + input_offsets1, mask=row_mask[:, None])
        tl.store(clone_ptr + clone_offsets1, values1, mask=row_mask[:, None])


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
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


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, tuple[int, int]]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor, got {type(x)!r}")
    if not x.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if x.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {x.dtype}")
    if x.ndim != 3:
        raise ValueError(f"{REPRO_ID} expects rank-3 input, got shape={tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(x.stride())}")

    batch, seq, hidden = (int(dim) for dim in x.shape)
    if seq != 1024 or hidden != 768:
        raise ValueError(f"unexpected captured shape {tuple(x.shape)}, expected [B, 1024, 768]")

    output_shape = _resolve_view_shape(shape_param, int(x.numel()))
    expected_shape = (seq * batch, hidden)
    if output_shape != expected_shape:
        raise ValueError(f"unexpected output shape {output_shape}, expected {expected_shape}")
    return x, output_shape


@oracle_impl(hardware="H100", shapes="(T([8, 1024, 768], f32), S([8192, 768]))")
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

    x, output_shape = _validate_inputs(inputs)
    batch, seq, hidden = (int(dim) for dim in x.shape)
    clone = torch.empty_strided(
        (seq, batch, hidden),
        (batch * hidden, hidden, 1),
        device=x.device,
        dtype=x.dtype,
    )
    total_rows = seq * batch
    grid = lambda meta: (triton.cdiv(total_rows, meta["BLOCK_ROWS"]),)
    _seq_batch_layout_kernel[grid](
        x,
        clone,
        TOTAL_ROWS=total_rows,
        B=batch,
        S=seq,
        H=hidden,
    )
    return clone.view(output_shape)


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
