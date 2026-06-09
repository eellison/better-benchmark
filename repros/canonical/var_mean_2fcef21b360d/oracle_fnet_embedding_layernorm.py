"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Google FNet embedding assembly and LayerNorm inference scope in one fixed-hidden Triton row kernel, including token embedding gathers, expanded token-type id gathers, positional embedding gathers, fp32 population var_mean over hidden dim 768, generated `libdevice.rsqrt(var + 1e-12)` lowering, affine epilogue, and final contiguous `[16384, 768]` view output, whereas Inductor lowers the decomposed embedding/expand/add/var_mean/rsqrt/affine/view graph through its generic persistent reduction schedule; Inductor cannot emit this dedicated form today because normalization template canonicalization does not recognize this broadcast token-type plus positional embedding producer pattern as a semantic embedding-LayerNorm lowering with fixed hidden size and direct indexed loads; the fix is NEW_PATTERN: add a guarded FNet embedding-LayerNorm template that folds the gathered embedding producers into a shape-specialized row-normalization kernel while preserving population variance and the generated rsqrt sequence."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None
    libdevice = None

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


BATCH = 32
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
BLOCK_H = 1024
WORD_VOCAB = 32000
TOKEN_TYPE_VOCAB = 4
POSITION_VOCAB = 512
EPS = 1.0e-12
CLASSIFICATION = "NEW_PATTERN"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None and libdevice is not None:

    @triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _fnet_embedding_layernorm_kernel(
        word_table_ptr,
        token_ids_ptr,
        token_type_ids_ptr,
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
        seq = rows % seq_len

        token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)
        token_type_id = tl.load(token_type_ids_ptr + seq, mask=row_mask, other=0)
        position_id = tl.load(position_ids_ptr + seq, mask=row_mask, other=0)

        word = tl.load(
            word_table_ptr + token_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        token_type = tl.load(
            token_type_table_ptr + token_type_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_table_ptr + position_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        x = (word + token_type) + position
        x_for_reduce = tl.where(mask, x, 0.0)
        sum_x = tl.sum(x_for_reduce, axis=1)[:, None].to(tl.float32)
        mean = sum_x / (tl.full([1, 1], 768, tl.int32)).to(tl.float32)

        centered_for_var = x - mean
        var_terms = centered_for_var * centered_for_var
        sum_var = tl.sum(tl.where(mask, var_terms, 0.0), axis=1)[:, None].to(tl.float32)
        variance = sum_var / tl.full([1, 1], 768.0, tl.float32)
        invstd = libdevice.rsqrt(variance + eps)

        centered = x - mean
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
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        word_table,
        token_ids,
        token_type_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        token_type_expand_shape,
        output_shape,
    ) = inputs

    tensor_inputs = (
        word_table,
        token_ids,
        token_type_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first eight repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (
        (WORD_VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (1, SEQ_LEN),
        (TOKEN_TYPE_VOCAB, HIDDEN),
        (POSITION_VOCAB, HIDDEN),
        (1, SEQ_LEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")

    index_tensors = (token_ids, token_type_ids, position_ids)
    if not all(value.dtype == torch.int64 for value in index_tensors):
        raise TypeError("token ids, token-type ids, and position ids must be torch.int64")
    fp32_tensors = (word_table, token_type_table, position_table, weight, bias)
    if not all(value.dtype == torch.float32 for value in fp32_tensors):
        raise TypeError("embedding tables, layernorm weight, and bias must be torch.float32")

    token_type_expand_shape_tuple = _shape_tuple(token_type_expand_shape)
    output_shape_tuple = _shape_tuple(output_shape)
    if token_type_expand_shape_tuple != (BATCH, SEQ_LEN):
        raise ValueError(f"unexpected token-type expand shape: {token_type_expand_shape!r}")
    if output_shape_tuple != (ROWS, HIDDEN):
        raise ValueError(f"unexpected output view shape: {output_shape!r}")

    return (
        word_table,
        token_ids,
        token_type_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        token_type_expand_shape_tuple,
        output_shape_tuple,
    )


def oracle_forward(inputs):
    """Run the complete Repro.forward embedding, LayerNorm, affine, and view scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same single f32 `[16384, 768]` output with contiguous `(768, 1)` stride.
    """
    if triton is None or libdevice is None:
        raise RuntimeError("Triton and Inductor libdevice helpers are required for this oracle")

    (
        word_table,
        token_ids,
        token_type_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        _token_type_expand_shape,
        output_shape,
    ) = _validate_inputs(inputs)

    out = torch.empty_strided(
        output_shape,
        (HIDDEN, 1),
        device=word_table.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["XBLOCK"]),)
    _fnet_embedding_layernorm_kernel[grid](
        word_table,
        token_ids,
        token_type_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        block_h=BLOCK_H,
        rows_total=ROWS,
    )
    return out


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
