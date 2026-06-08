"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Longformer layout chain by directly materializing the required fresh contiguous `[B*T, H*D]` output from the original contiguous `[B*H*(T/S), S, D]` input with one shape-specialized Triton layout copy, whereas Inductor already reaches the same CUDAGraph-measured mandatory read/write traffic envelope for this captured view/permute/clone/view scope; Inductor cannot materially improve this isolated repro with scheduler fusion, scatter/reduce, split-K, algebraic elimination, or recomputation because the user-visible result is a non-aliasing contiguous clone of all elements; the fix is BANDWIDTH_BOUND: record it as at floor unless broader layout-copy bandwidth, launch, or allocation improvements move both paths."""
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
BLOCK_ROWS = 64
CLASSIFICATION = "BANDWIDTH_BOUND"

if triton is not None:

    @triton.jit
    def _longformer_layout_chain_kernel(
        x,
        out,
        out_rows: tl.constexpr,
        out_cols: tl.constexpr,
        seq: tl.constexpr,
        tokens: tl.constexpr,
        heads: tl.constexpr,
        dim: tl.constexpr,
        chunks: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        row_mask = rows < out_rows

        batch = rows // tokens
        token = rows - batch * tokens
        chunk = token // seq
        seq_index = token - chunk * seq

        cols0 = tl.arange(0, 512)
        head0 = cols0 // dim
        dim0 = cols0 - head0 * dim
        in0 = (
            ((batch[:, None] * heads * chunks + head0[None, :] * chunks + chunk[:, None])
             * seq + seq_index[:, None])
            * dim + dim0[None, :]
        )
        out0 = rows[:, None] * out_cols + cols0[None, :]
        values0 = tl.load(x + in0, mask=row_mask[:, None])
        tl.store(out + out0, values0, mask=row_mask[:, None])

        cols1 = 512 + tl.arange(0, 256)
        head1 = cols1 // dim
        dim1 = cols1 - head1 * dim
        in1 = (
            ((batch[:, None] * heads * chunks + head1[None, :] * chunks + chunk[:, None])
             * seq + seq_index[:, None])
            * dim + dim1[None, :]
        )
        out1 = rows[:, None] * out_cols + cols1[None, :]
        values1 = tl.load(x + in1, mask=row_mask[:, None])
        tl.store(out + out1, values1, mask=row_mask[:, None])


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_shape(value: Any, numel: int) -> tuple[int, ...]:
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


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int], int, int, int, int, int, int]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    x, shape0_raw, shape1_raw, shape2_raw, shape3_raw, shape4_raw = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(x)!r}")
    if not x.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA tensor input")
    if x.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {x.dtype}")
    if x.ndim != 3:
        raise ValueError(f"expected rank-3 input, got shape={tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(x.stride())}")
    if x.storage_offset() != 0:
        raise ValueError(f"expected zero input storage offset, got {x.storage_offset()}")

    numel = int(x.numel())
    input_rows, seq, dim = (int(v) for v in x.shape)
    shape0 = _resolve_shape(shape0_raw, numel)
    shape1 = _resolve_shape(shape1_raw, numel)
    shape2 = _resolve_shape(shape2_raw, numel)
    shape3 = _resolve_shape(shape3_raw, numel)
    output_shape = _resolve_shape(shape4_raw, numel)

    if len(shape0) != 5 or shape0[3] != 1:
        raise ValueError(f"unexpected first view shape: {shape0}")
    if len(shape1) != 4 or shape1 != (shape0[0], shape0[1], shape0[2], shape0[4]):
        raise ValueError(f"unexpected squeezed view shape: {shape1}")
    if len(shape2) != 4:
        raise ValueError(f"unexpected attention view shape: {shape2}")

    batch, heads, tokens, view_dim = shape2
    if view_dim != dim:
        raise ValueError(f"shape dim {view_dim} does not match input dim {dim}")
    if heads != 12 or dim != 64:
        raise ValueError(f"kernel specialization expects 12 heads of dim 64, got {heads}x{dim}")
    if tokens % seq != 0:
        raise ValueError(f"tokens={tokens} must be divisible by seq={seq}")
    chunks = tokens // seq
    if input_rows != batch * heads * chunks:
        raise ValueError(
            f"input rows {input_rows} do not match batch*heads*chunks "
            f"{batch}*{heads}*{chunks}"
        )
    if shape0 != (batch * heads, chunks, seq, 1, dim):
        raise ValueError(f"unexpected unsqueezed view shape: {shape0}")

    out_cols = heads * dim
    if shape3 != (tokens, batch, out_cols):
        raise ValueError(f"unexpected intermediate output view shape: {shape3}")
    if output_shape != (batch * tokens, out_cols):
        raise ValueError(f"unexpected final output shape: {output_shape}")

    return x, output_shape, batch, heads, tokens, dim, seq, chunks


def oracle_forward(inputs):
    """Run the full-scope two-clone Longformer layout materialization."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, output_shape, _batch, heads, tokens, dim, seq, chunks = _validate_inputs(inputs)
    out = torch.empty_strided(
        output_shape,
        (output_shape[1], 1),
        device=x.device,
        dtype=x.dtype,
    )
    _longformer_layout_chain_kernel[(triton.cdiv(output_shape[0], BLOCK_ROWS),)](
        x,
        out,
        out_rows=output_shape[0],
        out_cols=output_shape[1],
        seq=seq,
        tokens=tokens,
        heads=heads,
        dim=dim,
        chunks=chunks,
        block_rows=BLOCK_ROWS,
        num_warps=8,
    )
    return out


def _check_full_layout(instance, inputs) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()
    ok = (
        isinstance(actual, torch.Tensor)
        and tuple(actual.shape) == tuple(expected.shape)
        and actual.dtype == expected.dtype
        and tuple(actual.stride()) == tuple(expected.stride())
        and actual.storage_offset() == expected.storage_offset()
        and actual.untyped_storage().data_ptr() != inputs[0].untyped_storage().data_ptr()
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={tuple(actual.stride())} "
        f"storage_offset={actual.storage_offset()})"
    )
    return ok


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
        ok = ok and _check_full_layout(instance, inputs)
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
