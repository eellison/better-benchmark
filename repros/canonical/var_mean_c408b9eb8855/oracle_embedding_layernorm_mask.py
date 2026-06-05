"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-2 token-plus-generated-position embedding and layernorm forward in one Triton row kernel, including token embedding, generated position embedding, fp16 add rounding, fp32 var_mean over hidden size 768/1280 with eps=1e-5, affine epilogue, fp16 output view, and the adjacent-position ne mask as an all-false output, whereas Inductor currently lowers the generated iota/embedding/add/var_mean/affine/view and adjacent-position cat/slice/sub/ne graph through generic embedding and reduction fusion plus separate boolean mask work; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize generated-position GPT-2 embedding-layernorm with the algebraically constant adjacent-position mask into a single fixed-hidden row normalization template; the fix is NEW_PATTERN: add a GPT-2 embedding-layernorm lowering that folds token and generated position indexed loads into the row kernel, preserves the fp16 view output, and emits or eliminates the constant adjacent-position mask."""
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


SEQ_LEN = 1024
VOCAB = 50257
EPS = 1.0e-5
SUPPORTED_HIDDEN = (768, 1280)
CLASSIFICATION = "NEW_PATTERN"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _embedding_layernorm_mask_kernel(
        word_table_ptr,
        token_ids_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        mask_out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        col_mask = cols < hidden

        token_id = tl.load(token_ids_ptr + row)
        token = tl.load(
            word_table_ptr + token_id * hidden + cols,
            mask=col_mask,
            other=0.0,
        )
        position = tl.load(
            position_table_ptr + row * hidden + cols,
            mask=col_mask,
            other=0.0,
        )

        # aten.add on fp16 embeddings materializes fp16 before var_mean promotes
        # to fp32.
        x = (token + position).to(tl.float16).to(tl.float32)
        x_for_reduce = tl.where(col_mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        var_terms = tl.where(col_mask, centered * centered, 0.0)
        variance = tl.sum(var_terms, axis=0) / hidden
        invstd = tl.rsqrt(tl.maximum(variance, 0.0) + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        out = centered * invstd * weight + bias

        tl.store(out_ptr + row * hidden + cols, out.to(tl.float16), mask=col_mask)

        # The generated positions are [-1, 0, 1, ...] versus [0, 1, 2, ...],
        # so every adjacent difference equals one and ne(1) is always false.
        tl.store(mask_out_ptr + row, 0)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    int,
    int,
]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    word_table, token_ids, position_table, weight, bias, out_shape = inputs
    tensor_inputs = (word_table, token_ids, position_table, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    hidden = int(word_table.shape[1])
    if hidden not in SUPPORTED_HIDDEN:
        raise ValueError(f"hidden size {hidden} is not supported; expected one of {SUPPORTED_HIDDEN}")

    expected_shapes = (
        (VOCAB, hidden),
        (1, SEQ_LEN),
        (SEQ_LEN, hidden),
        (hidden,),
        (hidden,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if word_table.dtype != torch.float16 or position_table.dtype != torch.float16:
        raise TypeError("embedding tables must be torch.float16")
    if weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise TypeError("layernorm weight and bias must be torch.float16")
    if token_ids.dtype != torch.int64:
        raise TypeError(f"token ids must be torch.int64, got {token_ids.dtype}")

    out_shape_tuple = _shape_tuple(out_shape)
    if len(out_shape_tuple) != 2 or out_shape_tuple[1] != hidden or out_shape_tuple[0] not in (-1, SEQ_LEN):
        raise ValueError(f"unexpected output view shape parameter: {out_shape!r}")

    block_h = _next_power_of_2(hidden)
    return word_table, token_ids, position_table, weight, bias, out_shape_tuple, hidden, block_h


def oracle_embedding_layernorm_mask(
    word_table: torch.Tensor,
    token_ids: torch.Tensor,
    position_table: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_shape: tuple[int, ...],
    hidden: int,
    block_h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the complete Repro.forward return with one Triton row kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_embedding_layernorm_mask.py")

    out = torch.empty_strided(
        (SEQ_LEN, hidden),
        (hidden, 1),
        device=word_table.device,
        dtype=torch.float16,
    )
    mask = torch.empty_strided(
        (1, SEQ_LEN),
        (SEQ_LEN, 1),
        device=word_table.device,
        dtype=torch.bool,
    )
    _embedding_layernorm_mask_kernel[(SEQ_LEN,)](
        word_table,
        token_ids,
        position_table,
        weight,
        bias,
        out,
        mask,
        hidden=hidden,
        eps=EPS,
        BLOCK_H=block_h,
        num_warps=4 if block_h <= 1024 else 8,
        num_stages=3,
    )
    return out.view(out_shape), mask


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
    return oracle_embedding_layernorm_mask(*_validate_inputs(inputs))


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
