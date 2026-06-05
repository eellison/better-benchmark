"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MegatronBERT word/token-type/position embedding assembly, Inductor-style dropout before fp32 hidden-size-1024 layernorm, affine, final [8192,1024] view, and rsqrt/1024 side output in one fixed-hidden Triton row kernel, whereas Inductor lowers the graph through generic embedding gathers, RNG dropout pointwise work, normalization-template reduction, affine epilogue, and view scheduling; Inductor cannot do this today because norm-template canonicalization does not recognize gathered embedding producers plus stochastic pre-normalization dropout and a live inverse-std side output as one semantic embedding-layernorm pattern; the fix is NEW_PATTERN: add a Megatron/BERT embedding-dropout-layernorm template that folds indexed embedding loads, Inductor RNG, fixed-K row reduction, affine epilogue, final view, and side-output store into specialized codegen. Exact stochastic value equality with eager is not established for the internally generated seed, so this is a structural diagnostic oracle rather than a true_floor proof."""
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
WORD_VOCAB = 29056
TOKEN_TYPES = 2
POSITION_VOCAB = 512
BASE_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
BASE_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
OUTPUT_SHAPE = (ROWS, HIDDEN)
INVSTD_SHAPE = (BATCH, SEQ_LEN, 1)
INVSTD_STRIDE = (SEQ_LEN, 1, 1)
EPS = 1.0e-12
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 49
SEED_INDEX = 0
SEED_LOW = -9223372036854775808
SEED_HIGH = 9223372036854775807
BLOCK_H = 1024

if triton is not None:

    @triton.jit
    def _embedding_dropout_layernorm_kernel(
        word_embedding_ptr,
        word_ids_ptr,
        token_type_embedding_ptr,
        position_embedding_ptr,
        position_ids_ptr,
        weight_ptr,
        bias_ptr,
        seeds_ptr,
        out_ptr,
        invstd_div_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        seed_index: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden

        seq = row % seq_len
        word_id = tl.load(word_ids_ptr + row)
        position_id = tl.load(position_ids_ptr + seq)

        word = tl.load(
            word_embedding_ptr + word_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        token_type = tl.load(
            token_type_embedding_ptr + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_embedding_ptr + position_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        x = tl.where(mask, word + token_type + position, 0.0)
        flat_offsets = row * hidden + cols
        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, flat_offsets.to(tl.uint32))
        dropped = tl.where(random > dropout_p, x * dropout_scale, 0.0)

        mean = tl.sum(tl.where(mask, dropped, 0.0), axis=0) / hidden
        centered = dropped - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + flat_offsets, y, mask=mask)
        tl.store(invstd_div_ptr + row, invstd / hidden)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
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
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    word_embedding = _require_tensor(
        "arg2_1",
        inputs[0],
        (WORD_VOCAB, HIDDEN),
        torch.float32,
    )
    word_ids = _require_tensor("arg1_1", inputs[1], (BATCH, SEQ_LEN), torch.int64)
    token_type_embedding = _require_tensor(
        "arg4_1",
        inputs[2],
        (TOKEN_TYPES, HIDDEN),
        torch.float32,
    )
    position_embedding = _require_tensor(
        "arg5_1",
        inputs[3],
        (POSITION_VOCAB, HIDDEN),
        torch.float32,
    )
    position_ids = _require_tensor("arg3_1", inputs[4], (1, SEQ_LEN), torch.int64)
    weight = _require_tensor("arg6_1", inputs[5], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg7_1", inputs[6], (HIDDEN,), torch.float32)
    output_shape = _shape_tuple(inputs[7])

    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape: {inputs[7]!r}")

    device = word_embedding.device
    tensor_inputs = (
        word_ids,
        token_type_embedding,
        position_embedding,
        position_ids,
        weight,
        bias,
    )
    if any(value.device != device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return (
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        position_ids,
        weight,
        bias,
        output_shape,
    )


def _make_inductor_seeds(device: torch.device) -> torch.Tensor:
    """Match compiled Inductor's lowering of prims.inductor_seeds.default(49)."""
    seeds = torch.empty_strided((SEED_COUNT,), (1,), device=device, dtype=torch.int64)
    torch.ops.aten.randint.low_out(SEED_LOW, SEED_HIGH, [SEED_COUNT], out=seeds)
    return seeds


def oracle_forward(inputs):
    """Run the complete embedding-dropout-layernorm repro computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same f32 [8192,1024] view plus f32 [16,512,1] invstd/1024 side output.
    """
    if triton is None:
        raise RuntimeError(
            "Triton is required for oracle_megatronbert_embedding_dropout_layernorm.py"
        )

    (
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        position_ids,
        weight,
        bias,
        output_shape,
    ) = _validate_inputs(inputs)

    out_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=word_embedding.device,
        dtype=torch.float32,
    )
    invstd_div = torch.empty_strided(
        INVSTD_SHAPE,
        INVSTD_STRIDE,
        device=word_embedding.device,
        dtype=torch.float32,
    )
    seeds = _make_inductor_seeds(word_embedding.device)

    _embedding_dropout_layernorm_kernel[(ROWS,)](
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        position_ids,
        weight,
        bias,
        seeds,
        out_base,
        invstd_div,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        seed_index=SEED_INDEX,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=3,
    )
    return (torch.ops.aten.view.default(out_base, output_shape), invstd_div)


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
