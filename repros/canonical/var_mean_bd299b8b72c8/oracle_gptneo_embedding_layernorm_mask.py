"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-Neo token embedding plus generated position embedding, hidden-size-2048 LayerNorm affine epilogue, three aliasing [4096,2048] views, and all-false adjacent-position mask in one Triton row kernel, whereas Inductor currently lowers the decomposed embedding/iota/expand/cat/var_mean/affine/view graph through generic indirect-index and normalization scheduling; Inductor cannot do this today because norm-template canonicalization does not recognize GPT-Neo generated-position embedding LayerNorm with sibling constant mask emission and multi-view alias returns as one semantic pattern; the fix is NEW_PATTERN: add a GPT-style embedding-LayerNorm template that folds token and position gathers, row reduction, affine stores, constant-mask epilogue, and alias-view returns into one lowering."""
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

BATCH = 32
SEQ_LEN = 128
ROWS = BATCH * SEQ_LEN
HIDDEN = 2048
TOKEN_VOCAB = 50257
POSITION_ROWS = 2048
EPS = 1.0e-5
BLOCK_H = 2048
DEFAULT_NUM_WARPS = 8
CLASSIFICATION = "NEW_PATTERN"

if triton is not None:

    @triton.jit
    def _gptneo_embedding_layernorm_mask_kernel(
        token_table_ptr,
        token_ids_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        mask_out_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)

        token_id = tl.load(token_ids_ptr + row)
        position_id = row % seq_len

        token = tl.load(token_table_ptr + token_id * hidden + cols).to(tl.float32)
        position = tl.load(position_table_ptr + position_id * hidden + cols).to(tl.float32)
        x = token + position

        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(centered * centered, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        bias = tl.load(bias_ptr + cols).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + row * hidden + cols, y)
        tl.store(mask_out_ptr + row, 0)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _matches_shape(actual: tuple[int, ...], expected: tuple[int, ...]) -> bool:
    return len(actual) == len(expected) and all(
        got == want or got == -1 or want == -1
        for got, want in zip(actual, expected)
    )


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects 9 inputs, got {len(inputs)}")

    (
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        shape0,
        shape1,
        expand_shape,
        shape3,
    ) = inputs
    tensor_inputs = (token_table, token_ids, position_table, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (
        (TOKEN_VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (POSITION_ROWS, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if token_table.dtype != torch.float32 or position_table.dtype != torch.float32:
        raise TypeError("embedding tables must be torch.float32")
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("LayerNorm weight and bias must be torch.float32")
    if token_ids.dtype != torch.int64:
        raise TypeError(f"token ids must be torch.int64, got {token_ids.dtype}")

    out_shapes = tuple(_shape_tuple(shape) for shape in (shape0, shape1, shape3))
    for index, shape in enumerate(out_shapes):
        if not _matches_shape(shape, (ROWS, HIDDEN)):
            raise ValueError(f"unexpected output shape {index}: {shape!r}")

    expand_shape_tuple = _shape_tuple(expand_shape)
    if not _matches_shape(expand_shape_tuple, (BATCH, SEQ_LEN)):
        raise ValueError(f"unexpected expand shape parameter: {expand_shape!r}")

    return token_table, token_ids, position_table, weight, bias, *out_shapes


def oracle_gptneo_embedding_layernorm_mask(
    token_table: torch.Tensor,
    token_ids: torch.Tensor,
    position_table: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    shape0: tuple[int, int],
    shape1: tuple[int, int],
    shape3: tuple[int, int],
    *,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the complete Repro.forward output contract."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_gptneo_embedding_layernorm_mask.py")

    base = torch.empty_strided(
        (BATCH, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=token_table.device,
        dtype=torch.float32,
    )
    mask = torch.empty_strided(
        (BATCH, SEQ_LEN),
        (SEQ_LEN, 1),
        device=token_table.device,
        dtype=torch.bool,
    )
    _gptneo_embedding_layernorm_mask_kernel[(ROWS,)](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        base,
        mask,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )

    return base.view(shape0), base.view(shape1), mask, base.view(shape3)


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
    return oracle_gptneo_embedding_layernorm_mask(*_validate_inputs(inputs))


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
