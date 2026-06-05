"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DeBERTa embedding-layernorm-dropout result in one fixed-hidden Triton row kernel, including word and position embedding gathers, fp32 hidden-size-1536 var_mean normalization, affine, Inductor-style dropout, and the final [4096, 1536] view, whereas Inductor currently lowers the graph through generic embedding gathers, row-reduction normalization, stochastic pointwise work, and a final view; Inductor cannot do this today because norm-template canonicalization does not recognize gathered embedding assembly plus stochastic dropout as one fixed-hidden semantic pattern; the fix is NEW_PATTERN: add an embedding-layernorm-dropout lowering that folds indexed producers, row reduction, affine/dropout epilogue, and view preservation into specialized codegen."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims.* RNG ops

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


# --- Oracle kernel(s) ---
BATCH = 8
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 1536
WORD_VOCAB = 128100
POSITION_VOCAB = 512
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
EPS = 1.0e-7
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 73
CLASSIFICATION = "NEW_PATTERN"

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1, "BLOCK_H": 2048}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 1, "BLOCK_H": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 2, "BLOCK_H": 2048}, num_warps=8, num_stages=3),
        ],
        key=["hidden", "seq_len"],
    )
    @triton.jit
    def _deberta_embedding_layernorm_dropout_kernel(
        word_embedding_ptr,
        word_ids_ptr,
        position_embedding_ptr,
        position_ids_ptr,
        weight_ptr,
        bias_ptr,
        seed_ptr,
        out_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_H)
        col_mask = cols < hidden

        seq = rows % seq_len
        word_id = tl.load(word_ids_ptr + rows)
        position_id = tl.load(position_ids_ptr + seq)

        word = tl.load(
            word_embedding_ptr + word_id[:, None] * hidden + cols[None, :],
            mask=col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_embedding_ptr + position_id[:, None] * hidden + cols[None, :],
            mask=col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        x = tl.where(col_mask[None, :], word + position, 0.0)
        mean = tl.sum(x, axis=1) / hidden
        sq_mean = tl.sum(x * x, axis=1) / hidden
        variance = sq_mean - mean * mean
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = (x - mean[:, None]) * invstd[:, None] * weight[None, :] + bias[None, :]

        offsets = rows[:, None] * hidden + cols[None, :]
        seed = tl.load(seed_ptr)
        random = tl.rand(seed, offsets.to(tl.uint32))
        dropped = tl.where(random > dropout_p, y, 0.0) * dropout_scale
        tl.store(out_ptr + offsets, dropped, mask=col_mask[None, :])


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    word_embedding, word_ids, position_embedding, position_ids, weight, bias, output_shape = inputs
    tensor_inputs = (word_embedding, word_ids, position_embedding, position_ids, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first six repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (
        (WORD_VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (POSITION_VOCAB, HIDDEN),
        (1, SEQ_LEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")

    if word_embedding.dtype != torch.float32 or position_embedding.dtype != torch.float32:
        raise TypeError("embedding tables must be torch.float32")
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("layernorm weight and bias must be torch.float32")
    if word_ids.dtype != torch.int64 or position_ids.dtype != torch.int64:
        raise TypeError("embedding index tensors must be torch.int64")

    output_shape_tuple = _shape_tuple(output_shape)
    if output_shape_tuple != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape: {output_shape!r}")

    return word_embedding, word_ids, position_embedding, position_ids, weight, bias, output_shape_tuple


def _make_inductor_seed(device: torch.device) -> torch.Tensor:
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=device)
    return torch.ops.prims.inductor_lookup_seed.default(seeds, 0)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete Repro.forward embedding, layernorm, dropout, and view scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_deberta_embedding_layernorm_dropout.py")

    word_embedding, word_ids, position_embedding, position_ids, weight, bias, output_shape = _validate_inputs(inputs)
    out = torch.empty_strided(
        (BATCH, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=word_embedding.device,
        dtype=torch.float32,
    )
    seed = _make_inductor_seed(word_embedding.device)

    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_M"]),)
    _deberta_embedding_layernorm_dropout_kernel[grid](
        word_embedding,
        word_ids,
        position_embedding,
        position_ids,
        weight,
        bias,
        seed,
        out,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
    )
    return torch.ops.aten.view.default(out, output_shape)


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if actual.is_cuda:
            torch.cuda.synchronize()

    ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and tuple(actual.shape) == tuple(expected.shape)
        and actual.stride() == expected.stride()
        and actual.dtype == expected.dtype
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected_shape={tuple(expected.shape)}, oracle_shape={tuple(actual.shape)}, "
        f"expected_stride={expected.stride()}, oracle_stride={actual.stride()}, "
        f"expected_dtype={expected.dtype}, oracle_dtype={actual.dtype})"
    )
    return ok


# --- CLI entry point ---
def main() -> None:
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
        ok = _check_layout(instance, inputs) and ok
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
