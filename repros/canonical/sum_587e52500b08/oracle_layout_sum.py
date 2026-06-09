"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full cat-view-permute-clone materialization and the sibling column sum by writing the clone backing buffer while also producing the first row-sum stage from the same source tile, whereas Inductor materializes the clone and then rereads that full buffer in separate reduction kernels; Inductor cannot do this today because layout-copy scheduling does not group a returned materialized view with a reduction consumer of the same buffer; the fix is SCHEDULER_FUSION: teach Inductor to fuse layout materialization with deterministic partial reductions for sibling outputs that share the same producer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# bench_oracle() owns timing so CUDAGraph, GPU locking, and interleaved
# oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


BLOCK_ROWS_MATERIALIZE = 64
BLOCK_COLS_MATERIALIZE = 64
BLOCK_COLS_FINAL = 32
PARTIAL_CHUNKS = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_view_shape(value: Any, numel: int) -> tuple[int, ...]:
    dims = list(_shape_tuple(value))
    inferred = dims.count(-1)
    if inferred > 1:
        raise ValueError(f"only one inferred dimension is valid, got {dims}")
    if inferred == 1:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"cannot infer shape {dims} for numel={numel}")
        dims[dims.index(-1)] = numel // known
    resolved = tuple(dims)
    if _numel(resolved) != numel:
        raise ValueError(f"shape {resolved} has {_numel(resolved)} elements, expected {numel}")
    return resolved


def _require_input_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if value.ndim != 4:
        raise ValueError(f"{name} must be rank-4, got shape={tuple(value.shape)}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int, int, int],
    tuple[int, int],
    tuple[int],
]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects seven inputs, got {len(inputs)}")

    x0 = _require_input_tensor("getitem_44", inputs[0])
    x1 = _require_input_tensor("getitem_45", inputs[1])
    x2 = _require_input_tensor("getitem_46", inputs[2])

    if x1.shape != x0.shape or x2.shape != x0.shape:
        raise ValueError(f"all tensor inputs must share shape, got {x0.shape}, {x1.shape}, {x2.shape}")
    if x1.stride() != x0.stride() or x2.stride() != x0.stride():
        raise ValueError(f"all tensor inputs must share stride, got {x0.stride()}, {x1.stride()}, {x2.stride()}")
    if x1.device != x0.device or x2.device != x0.device:
        raise ValueError("all tensor inputs must be on the same device")

    batch, heads, tokens, channels = (int(dim) for dim in x0.shape)
    input_numel = int(x0.numel())
    cat_numel = input_numel * 3
    cols = 3 * heads * channels
    rows = batch * tokens

    shape0 = _resolve_view_shape(inputs[3], cat_numel)
    shape1 = _resolve_view_shape(inputs[4], cat_numel)
    shape2 = _resolve_view_shape(inputs[5], cat_numel)
    shape3 = _resolve_view_shape(inputs[6], cols)

    expected_shape0 = (3, batch, heads, tokens, channels)
    expected_shape1 = (batch, tokens, cols)
    expected_shape2 = (rows, cols)
    expected_shape3 = (cols,)
    if shape0 != expected_shape0:
        raise ValueError(f"unexpected first view shape {shape0}, expected {expected_shape0}")
    if shape1 != expected_shape1:
        raise ValueError(f"unexpected second view shape {shape1}, expected {expected_shape1}")
    if shape2 != expected_shape2:
        raise ValueError(f"unexpected third view shape {shape2}, expected {expected_shape2}")
    if shape3 != expected_shape3:
        raise ValueError(f"unexpected final view shape {shape3}, expected {expected_shape3}")
    if rows % PARTIAL_CHUNKS != 0:
        raise ValueError(f"rows={rows} must divide evenly across {PARTIAL_CHUNKS} partial chunks")

    clone_shape = (batch, tokens, 3, heads, channels)
    return x0, x1, x2, clone_shape, shape2, shape3


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _materialize_partial_sum_kernel(
        x0_ptr,
        x1_ptr,
        x2_ptr,
        clone_ptr,
        partial_ptr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        heads: tl.constexpr,
        tokens: tl.constexpr,
        channels: tl.constexpr,
        stride_b: tl.constexpr,
        stride_h: tl.constexpr,
        stride_t: tl.constexpr,
        stride_c: tl.constexpr,
        chunk_rows: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        col_offsets = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
        source_id = col_offsets // (heads * channels)
        source_inner = col_offsets - source_id * (heads * channels)
        head_offsets = source_inner // channels
        channel_offsets = source_inner - head_offsets * channels
        col_mask = col_offsets < cols
        acc = tl.zeros((BLOCK_COLS,), tl.float32)

        for row_start in tl.range(0, chunk_rows, RBLOCK):
            row_in_chunk = row_start + tl.arange(0, RBLOCK)[:, None]
            row_offsets = tl.program_id(1) * chunk_rows + row_in_chunk
            row_mask = row_in_chunk < chunk_rows
            mask = row_mask & col_mask
            batch_offsets = row_offsets // tokens
            token_offsets = row_offsets - batch_offsets * tokens
            source_offsets = (
                batch_offsets * stride_b
                + head_offsets * stride_h
                + token_offsets * stride_t
                + channel_offsets * stride_c
            )

            vals0 = tl.load(x0_ptr + source_offsets, mask=mask & (source_id == 0), other=0.0)
            vals1 = tl.load(x1_ptr + source_offsets, mask=mask & (source_id == 1), other=0.0)
            vals2 = tl.load(x2_ptr + source_offsets, mask=mask & (source_id == 2), other=0.0)
            vals = tl.where(source_id == 0, vals0, tl.where(source_id == 1, vals1, vals2))

            clone_offsets = row_offsets * cols + col_offsets
            tl.store(clone_ptr + clone_offsets, vals, mask=mask)
            acc += tl.sum(tl.where(row_mask, vals, 0.0), axis=0)

        reduced = acc
        partial_offsets = tl.program_id(1) * cols + col_offsets
        tl.store(partial_ptr + partial_offsets, reduced[None, :], mask=col_mask)

    @triton.jit
    def _final_sum_kernel(
        partial_ptr,
        out_ptr,
        cols: tl.constexpr,
        chunks: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        col_vector = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        col_offsets = col_vector[:, None]
        chunk_offsets = tl.arange(0, RBLOCK)[None, :]
        mask = (col_offsets < cols) & (chunk_offsets < chunks)
        vals = tl.load(partial_ptr + chunk_offsets * cols + col_offsets, mask=mask, other=0.0)
        reduced = tl.sum(vals, axis=1)
        tl.store(out_ptr + col_vector, reduced, mask=col_vector < cols)


@oracle_impl(hardware="H100", shapes="(T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), S([3, 128, 12, 198, 64]), S([128, 198, 2304]), S([25344, 2304]), S([2304]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x0, x1, x2, clone_shape, matrix_shape, sum_shape = _validate_inputs(inputs)
    batch, tokens, _, heads, channels = clone_shape
    rows, cols = matrix_shape
    chunk_rows = rows // PARTIAL_CHUNKS
    clone = torch.empty(clone_shape, device=x0.device, dtype=x0.dtype)
    partial = torch.empty((PARTIAL_CHUNKS, cols), device=x0.device, dtype=torch.float32)
    sum_keepdim = torch.empty((1, cols), device=x0.device, dtype=torch.float32)

    grid = (triton.cdiv(cols, BLOCK_COLS_MATERIALIZE), PARTIAL_CHUNKS)
    _materialize_partial_sum_kernel[grid](
        x0,
        x1,
        x2,
        clone,
        partial,
        rows,
        cols,
        heads,
        tokens,
        channels,
        x0.stride(0),
        x0.stride(1),
        x0.stride(2),
        x0.stride(3),
        chunk_rows,
        BLOCK_COLS=BLOCK_COLS_MATERIALIZE,
        RBLOCK=BLOCK_ROWS_MATERIALIZE,
        num_warps=2,
    )

    final_rblock = triton.next_power_of_2(PARTIAL_CHUNKS)
    _final_sum_kernel[(triton.cdiv(cols, BLOCK_COLS_FINAL),)](
        partial,
        sum_keepdim,
        cols,
        PARTIAL_CHUNKS,
        BLOCK_COLS=BLOCK_COLS_FINAL,
        RBLOCK=final_rblock,
        num_warps=2,
    )

    matrix = clone.view(matrix_shape)
    return matrix.permute(1, 0), sum_keepdim.view(sum_shape)


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
