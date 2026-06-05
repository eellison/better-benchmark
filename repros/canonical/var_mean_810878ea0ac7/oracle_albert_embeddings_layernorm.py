"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ALBERT embedding composition and LayerNorm scope in one fixed-hidden Triton row kernel, including token embedding gathers, gathered token-type ids, token-type and position embedding gathers, fp32 var_mean over hidden dim 128, affine epilogue, and final contiguous [4096,128] view, whereas the best Inductor configuration is already within measurement noise of this full-scope specialized implementation on the measured shape; Inductor cannot get a confirmed local win here because the remaining cost is dominated by required indexed embedding reads, hidden-dim reduction math, output stores, and launch overhead, and the oracle does not establish a faster floor; the fix is BANDWIDTH_BOUND: mark this repro at floor and only revisit if a broader embedding-layernorm lowering beats the current compiled kernel on the exact full scope."""
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


ROWS = 8 * 512
BATCH = 8
SEQ_LEN = 512
HIDDEN = 128
TOKEN_VOCAB = 30000
TYPE_VOCAB = 2
EPS = 1.0e-12


if triton is not None:

    @triton.jit
    def _albert_embedding_layernorm_kernel(
        token_table_ptr,
        token_ids_ptr,
        token_type_source_ptr,
        position_ids_ptr,
        type_table_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        eps: tl.constexpr,
    ):
        row0 = tl.program_id(0) * 2
        row1 = row0 + 1
        cols = tl.arange(0, hidden)

        token_id0 = tl.load(token_ids_ptr + row0)
        position_id0 = tl.load(position_ids_ptr + (row0 % seq_len))
        token_type_id0 = tl.load(token_type_source_ptr + position_id0)

        token0 = tl.load(token_table_ptr + token_id0 * hidden + cols).to(tl.float32)
        token_type0 = tl.load(type_table_ptr + token_type_id0 * hidden + cols).to(tl.float32)
        position0 = tl.load(position_table_ptr + position_id0 * hidden + cols).to(tl.float32)
        x0 = token0 + token_type0 + position0

        mean0 = tl.sum(x0, axis=0) / hidden
        centered0 = x0 - mean0
        variance0 = tl.sum(centered0 * centered0, axis=0) / hidden
        invstd0 = tl.rsqrt(variance0 + eps)

        token_id1 = tl.load(token_ids_ptr + row1)
        position_id1 = tl.load(position_ids_ptr + (row1 % seq_len))
        token_type_id1 = tl.load(token_type_source_ptr + position_id1)

        token1 = tl.load(token_table_ptr + token_id1 * hidden + cols).to(tl.float32)
        token_type1 = tl.load(type_table_ptr + token_type_id1 * hidden + cols).to(tl.float32)
        position1 = tl.load(position_table_ptr + position_id1 * hidden + cols).to(tl.float32)
        x1 = token1 + token_type1 + position1

        mean1 = tl.sum(x1, axis=0) / hidden
        centered1 = x1 - mean1
        variance1 = tl.sum(centered1 * centered1, axis=0) / hidden
        invstd1 = tl.rsqrt(variance1 + eps)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        bias = tl.load(bias_ptr + cols).to(tl.float32)
        out0 = centered0 * invstd0 * weight + bias
        out1 = centered1 * invstd1 * weight + bias

        tl.store(out_ptr + row0 * hidden + cols, out0)
        tl.store(out_ptr + row1 * hidden + cols, out1)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
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
        token_table,
        token_ids,
        token_type_source,
        position_ids,
        type_table,
        position_table,
        weight,
        bias,
        expand_shape,
        output_shape,
    ) = inputs

    tensor_inputs = (
        token_table,
        token_ids,
        token_type_source,
        position_ids,
        type_table,
        position_table,
        weight,
        bias,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first eight repro inputs must be tensors")

    expected_shapes = (
        (TOKEN_VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (1, SEQ_LEN),
        (1, SEQ_LEN),
        (TYPE_VOCAB, HIDDEN),
        (SEQ_LEN, HIDDEN),
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

    index_tensors = (token_ids, token_type_source, position_ids)
    if not all(value.dtype == torch.int64 for value in index_tensors):
        raise TypeError("token ids, token-type source, and position ids must be torch.int64")
    fp32_tensors = (token_table, type_table, position_table, weight, bias)
    if not all(value.dtype == torch.float32 for value in fp32_tensors):
        raise TypeError("embedding tables, layernorm weight, and bias must be torch.float32")

    expand_shape_tuple = _shape_tuple(expand_shape)
    output_shape_tuple = _shape_tuple(output_shape)
    if expand_shape_tuple != (BATCH, SEQ_LEN):
        raise ValueError(f"unexpected expand shape parameter: {expand_shape!r}")
    if output_shape_tuple != (ROWS, HIDDEN):
        raise ValueError(f"unexpected output shape parameter: {output_shape!r}")

    return (
        token_table,
        token_ids,
        token_type_source,
        position_ids,
        type_table,
        position_table,
        weight,
        bias,
        expand_shape_tuple,
        output_shape_tuple,
    )


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
        raise RuntimeError("Triton is required for oracle_albert_embeddings_layernorm.py")

    (
        token_table,
        token_ids,
        token_type_source,
        position_ids,
        type_table,
        position_table,
        weight,
        bias,
        _expand_shape,
        output_shape,
    ) = _validate_inputs(inputs)

    out = torch.empty_strided(
        output_shape,
        (HIDDEN, 1),
        device=token_table.device,
        dtype=torch.float32,
    )
    _albert_embedding_layernorm_kernel[(ROWS // 2,)](
        token_table,
        token_ids,
        token_type_source,
        position_ids,
        type_table,
        position_table,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        num_warps=1,
        num_stages=3,
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
