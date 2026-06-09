"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full Moondream fp16 token embedding gather, fp32 hidden-size-2048 row layernorm, fp16 affine epilogue cast back to fp16, and four returned `[512,2048]` views from one result storage in a single shape-specialized Triton row kernel, whereas Inductor lowers the embedding producer, fp32 var_mean/layernorm, dtype conversions, affine, and multi-view alias returns through generic decomposition and normalization scheduling; Inductor cannot do this today because norm-template canonicalization does not recognize direct fp16 embedding gather feeding fixed-K fp32 layernorm with aliased view outputs as one semantic embedding-layernorm pattern; the fix is NEW_PATTERN: add an embedding-layernorm alias template that folds indexed embedding loads, fp32 row reduction, affine epilogue, fp16 output cast, and aliased view returns into one specialized lowering."""
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
ROWS = 512
HIDDEN = 2048
VOCAB = 51200
EPS = 1.0e-5
BLOCK_H = 2048
DEFAULT_NUM_WARPS = 8
CLASSIFICATION = "NEW_PATTERN"

if triton is not None:

    @triton.jit
    def _embedding_layernorm_kernel(
        embedding_ptr,
        token_ids_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)

        token_id = tl.load(token_ids_ptr + row)
        x = tl.load(embedding_ptr + token_id * hidden + cols).to(tl.float32)

        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(centered * centered, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        bias = tl.load(bias_ptr + cols).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + row * hidden + cols, y)


def _shape_tuple(shape: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    embedding, token_ids, weight, bias, shape0, shape1, shape2, shape3 = inputs
    tensor_inputs = (embedding, token_ids, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = (
        (VOCAB, HIDDEN),
        (1, ROWS),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if embedding.dtype != torch.float16:
        raise TypeError(f"embedding table must be f16, got {embedding.dtype}")
    if weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise TypeError("layernorm weight and bias must be f16")
    if token_ids.dtype != torch.int64:
        raise TypeError(f"token ids must be i64, got {token_ids.dtype}")

    out_shapes = tuple(_shape_tuple(shape) for shape in (shape0, shape1, shape2, shape3))
    for index, shape in enumerate(out_shapes, start=4):
        if shape != (ROWS, HIDDEN):
            raise ValueError(f"input {index} unexpected output shape parameter: {shape!r}")

    return embedding, token_ids, weight, bias, *out_shapes


def oracle_embedding_layernorm_aliases(
    embedding: torch.Tensor,
    token_ids: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    shape0: tuple[int, int],
    shape1: tuple[int, int],
    shape2: tuple[int, int],
    shape3: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the full Repro.forward result and return four aliased views."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    base = torch.empty_strided(
        (1, ROWS, HIDDEN),
        (ROWS * HIDDEN, HIDDEN, 1),
        device=embedding.device,
        dtype=embedding.dtype,
    )
    _embedding_layernorm_kernel[(ROWS,)](
        embedding,
        token_ids,
        weight,
        bias,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=4,
    )

    return (
        base.view(shape0),
        base.view(shape1),
        base.view(shape2),
        base.view(shape3),
    )


@oracle_impl(hardware="H100", shapes="(T([51200, 2048], f16), T([1, 512], i64, gen=Index(51200)), T([2048], f16), T([2048], f16), S([512, 2048]), S([512, 2048]), S([512, 2048]), S([512, 2048]))")
def oracle_forward(inputs):
    """Run the full-scope Moondream embedding + LayerNorm oracle."""
    return oracle_embedding_layernorm_aliases(*_validate_inputs(inputs))


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
