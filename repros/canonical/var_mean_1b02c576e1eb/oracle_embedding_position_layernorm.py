"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full GPT-Neo token embedding gather, fixed position embedding gather, deterministic all-false position-difference mask, fp32 hidden-size-2048 LayerNorm var_mean(correction=0)+rsqrt eps=1e-5, affine scale/bias, and final [4096,2048] view in one shape-specialized Triton row kernel, whereas Inductor currently lowers the embedding/position producers, var_mean reduction, mask side output, and LayerNorm epilogue through general gather/reduction/pointwise scheduling; Inductor cannot do this today because norm-template canonicalization does not recognize token and positional embedding gathers feeding fixed-K LayerNorm with an independent small boolean side output as one semantic inference pattern; the fix is NEW_PATTERN: add an embedding-position-LayerNorm template that folds indexed loads, row mean/variance reduction, affine epilogue, view return, and deterministic mask materialization into one specialized lowering."""
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
SEQ = 128
ROWS = BATCH * SEQ
HIDDEN = 2048
VOCAB = 50_257
POSITION_ROWS = 2048
INPUT_ID_SHAPE = (BATCH, SEQ)
TOKEN_EMBEDDING_SHAPE = (VOCAB, HIDDEN)
POSITION_EMBEDDING_SHAPE = (POSITION_ROWS, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
MASK_SHAPE = (BATCH, SEQ)
MASK_STRIDE = (SEQ, 1)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
EXPAND_SHAPE_PARAM = (BATCH, -1)
EPS = 1.0e-5
BLOCK_H = HIDDEN

if triton is not None:

    @triton.jit
    def _embedding_position_layernorm_kernel(
        token_embedding_ptr,
        input_ids_ptr,
        position_embedding_ptr,
        weight_ptr,
        bias_ptr,
        mask_out_ptr,
        out_ptr,
        hidden: tl.constexpr,
        seq: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        offsets = row * hidden + cols
        seq_index = row % seq

        token_id = tl.load(input_ids_ptr + row)
        token = tl.load(
            token_embedding_ptr + token_id * hidden + cols,
            eviction_policy="evict_first",
        ).to(tl.float32)
        position = tl.load(
            position_embedding_ptr + seq_index * hidden + cols,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = token + position

        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(centered * centered, axis=0) / hidden
        inv_std = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, eviction_policy="evict_first").to(tl.float32)
        bias = tl.load(bias_ptr + cols, eviction_policy="evict_first").to(tl.float32)
        y = centered * inv_std * weight + bias

        tl.store(out_ptr + offsets, y)
        tl.store(mask_out_ptr + row, False)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    (
        token_embedding,
        input_ids,
        position_embedding,
        weight,
        bias,
        expand_shape,
        output_shape,
    ) = inputs

    token_embedding_t = _require_tensor(
        "arg1_1", token_embedding, TOKEN_EMBEDDING_SHAPE, torch.float32
    )
    input_ids_t = _require_tensor("arg0_1", input_ids, INPUT_ID_SHAPE, torch.int64)
    position_embedding_t = _require_tensor(
        "arg2_1", position_embedding, POSITION_EMBEDDING_SHAPE, torch.float32
    )
    weight_t = _require_tensor("arg3_1", weight, AFFINE_SHAPE, torch.float32)
    bias_t = _require_tensor("arg4_1", bias, AFFINE_SHAPE, torch.float32)

    device = token_embedding_t.device
    tensor_inputs = (input_ids_t, position_embedding_t, weight_t, bias_t)
    if any(value.device != device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    if _shape_tuple(expand_shape) != EXPAND_SHAPE_PARAM:
        raise ValueError(f"unexpected expand shape parameter: {expand_shape!r}")
    out_shape = _shape_tuple(output_shape)
    if out_shape != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {output_shape!r}")

    return (
        token_embedding_t,
        input_ids_t,
        position_embedding_t,
        weight_t,
        bias_t,
        out_shape,
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
        raise RuntimeError("Triton is required for oracle_embedding_position_layernorm.py")

    (
        token_embedding,
        input_ids,
        position_embedding,
        weight,
        bias,
        output_shape,
    ) = _validate_inputs(inputs)

    mask_out = torch.empty_strided(
        MASK_SHAPE,
        MASK_STRIDE,
        device=token_embedding.device,
        dtype=torch.bool,
    )
    output = torch.empty_strided(
        output_shape,
        OUTPUT_STRIDE,
        device=token_embedding.device,
        dtype=torch.float32,
    )

    _embedding_position_layernorm_kernel[(ROWS,)](
        token_embedding,
        input_ids,
        position_embedding,
        weight,
        bias,
        mask_out,
        output,
        hidden=HIDDEN,
        seq=SEQ,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=8,
        num_stages=3,
    )
    return (mask_out, output)


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
