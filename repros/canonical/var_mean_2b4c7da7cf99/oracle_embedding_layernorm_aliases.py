"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full MegatronBERT embedding composition, hidden-size-1024 var_mean layernorm, affine epilogue, and three aliasing `[8192,1024]` view returns in one shape-specialized Triton row kernel, whereas Inductor currently lowers the token gather, constant token-type embedding, position gather, var_mean normalization, affine pointwise epilogue, and aliasing view returns through generic embedding and normalization schedules; Inductor cannot do this today because norm-template canonicalization does not recognize a multi-producer embedding gather feeding fixed-K layernorm with shared-storage view aliases as one semantic embedding-layernorm pattern; the fix is NEW_PATTERN: add a guarded embedding-layernorm lowering that folds token, token-type, and position embedding loads with fp32 row reduction, affine storage, and aliased view returns into one specialized codegen path."""
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
BATCH = 16
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 1024
TOKEN_VOCAB = 29056
TOKEN_TYPE_ROWS = 2
POSITION_ROWS = 512
EPS = 1.0e-12
BLOCK_H = 1024
CLASSIFICATION = "NEW_PATTERN"

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"XBLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 2}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _megatron_embedding_layernorm_kernel(
        token_table_ptr,
        token_ids_ptr,
        token_type_table_ptr,
        position_table_ptr,
        position_ids_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        rows_total: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = rows < rows_total
        col_mask = cols < hidden
        mask = row_mask & col_mask

        token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)
        position_id = tl.load(position_ids_ptr + (rows % seq_len), mask=row_mask, other=0)

        token = tl.load(
            token_table_ptr + token_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        token_type = tl.load(
            token_type_table_ptr + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_table_ptr + position_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        x = tl.where(mask, token + token_type + position, 0.0)
        mean = tl.sum(x, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + rows * hidden + cols, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        token_table,
        token_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        out_shape0,
        out_shape1,
        out_shape2,
    ) = inputs

    tensor_inputs = (
        token_table,
        token_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first seven repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (
        (TOKEN_VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (TOKEN_TYPE_ROWS, HIDDEN),
        (POSITION_ROWS, HIDDEN),
        (1, SEQ_LEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")

    fp32_tensors = (token_table, token_type_table, position_table, weight, bias)
    if not all(value.dtype == torch.float32 for value in fp32_tensors):
        raise TypeError("embedding tables, LayerNorm weight, and bias must be torch.float32")
    if token_ids.dtype != torch.int64 or position_ids.dtype != torch.int64:
        raise TypeError("token ids and position ids must be torch.int64")

    output_shapes = tuple(_shape_tuple(shape) for shape in (out_shape0, out_shape1, out_shape2))
    for index, shape in enumerate(output_shapes, start=7):
        if shape != (ROWS, HIDDEN):
            raise ValueError(f"input {index} unexpected output shape parameter: {shape!r}")

    return (
        token_table,
        token_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        *output_shapes,
    )


def oracle_forward(inputs):
    """Run the full-scope MegatronBERT embedding + LayerNorm computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_embedding_layernorm_aliases.py")

    (
        token_table,
        token_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        out_shape0,
        out_shape1,
        out_shape2,
    ) = _validate_inputs(inputs)

    out_storage = torch.empty_strided(
        (BATCH, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=token_table.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["XBLOCK"]),)
    _megatron_embedding_layernorm_kernel[grid](
        token_table,
        token_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        out_storage,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        block_h=BLOCK_H,
        rows_total=ROWS,
    )

    return (
        out_storage.view(out_shape0),
        out_storage.view(out_shape1),
        out_storage.view(out_shape2),
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
