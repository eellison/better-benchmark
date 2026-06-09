"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer token/global/position embedding assembly and fp32 hidden-size-768 LayerNorm scope in one shape-specialized Triton row kernel, including the captured f16 embedding-add rounding, mask-derived position ids, affine scale/bias, and final f16 [1,4096,768] output, whereas Inductor lowers the decomposed embedding/position-id/add/var_mean/affine graph through generic gather and normalization scheduling; Inductor cannot emit this exact floor today because norm-template canonicalization does not recognize Longformer computed-position embedding producers feeding fixed-K f16-output LayerNorm as one semantic template; the fix is NEW_PATTERN: add a Longformer embedding-position-LayerNorm lowering that folds indexed loads, position-id construction, row reduction, affine epilogue, and f16 store into specialized codegen."""
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


# --- Oracle kernel(s) ---

BATCH = 1
SEQ_LEN = 4096
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
WORD_VOCAB = 50_265
POSITION_VOCAB = 4098
GLOBAL_VOCAB = 1
EPS = 1.0e-5
BLOCK_H = 1024
CLASSIFICATION = "NEW_PATTERN"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 1}, num_warps=2, num_stages=3),
            triton.Config({"XBLOCK": 2}, num_warps=2, num_stages=3),
            triton.Config({"XBLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=2),
        ],
        key=[],
    )
    @triton.jit
    def _longformer_embedding_position_layernorm_kernel(
        word_embedding_ptr,
        word_ids_ptr,
        cumsum_ptr,
        position_mask_ptr,
        position_embedding_ptr,
        global_embedding_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
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

        word_id = tl.load(word_ids_ptr + rows, mask=row_mask, other=0)
        cumsum_i32 = tl.load(cumsum_ptr + rows, mask=row_mask, other=0).to(tl.int32)
        position_mask = tl.load(position_mask_ptr + rows, mask=row_mask, other=0)
        position_id = (cumsum_i32 * position_mask).to(tl.int64) + 1

        word = tl.load(
            word_embedding_ptr + word_id * hidden + cols,
            mask=mask,
            other=0.0,
        )
        position = tl.load(
            position_embedding_ptr + position_id * hidden + cols,
            mask=mask,
            other=0.0,
        )
        global_token = tl.load(
            global_embedding_ptr + cols,
            mask=col_mask,
            other=0.0,
        )

        # The captured graph rounds after each f16 embedding add before the
        # fp32 LayerNorm reduction.
        pair_sum = (word + position).to(tl.float16)
        summed = (pair_sum + global_token).to(tl.float16).to(tl.float32)

        x = tl.where(mask, summed, 0.0)
        mean = tl.sum(x, axis=1)[:, None] / hidden
        centered = summed - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + rows * hidden + cols, y, mask=mask)


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
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    word_embedding = _require_tensor(
        "arg1_1", inputs[0], (WORD_VOCAB, HIDDEN), torch.float16
    )
    word_ids = _require_tensor("arg0_1", inputs[1], (BATCH, SEQ_LEN), torch.int64)
    cumsum = _require_tensor("cumsum", inputs[2], (BATCH, SEQ_LEN), torch.int64)
    position_mask = _require_tensor(
        "convert_element_type_1", inputs[3], (BATCH, SEQ_LEN), torch.int32
    )
    position_embedding = _require_tensor(
        "arg2_1", inputs[4], (POSITION_VOCAB, HIDDEN), torch.float16
    )
    global_embedding = _require_tensor(
        "arg3_1", inputs[5], (GLOBAL_VOCAB, HIDDEN), torch.float16
    )
    weight = _require_tensor("arg4_1", inputs[6], (HIDDEN,), torch.float16)
    bias = _require_tensor("arg5_1", inputs[7], (HIDDEN,), torch.float16)

    device = word_embedding.device
    tensor_inputs = (
        word_ids,
        cumsum,
        position_mask,
        position_embedding,
        global_embedding,
        weight,
        bias,
    )
    if any(value.device != device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return (
        word_embedding,
        word_ids,
        cumsum,
        position_mask,
        position_embedding,
        global_embedding,
        weight,
        bias,
    )


@oracle_impl(hardware="H100", shapes="(T([50265, 768], f16), T([1, 4096], i64, gen=Index(50265)), T([1, 4096], i64, gen=Index(4097)), T([1, 4096], i32, gen=Index(2)), T([4098, 768], f16), T([1, 768], f16), T([768], f16), T([768], f16))")
def oracle_forward(inputs):
    """Run the complete Repro.forward Longformer embedding + LayerNorm computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError(
            "Triton is required for oracle_longformer_embedding_position_layernorm.py"
        )

    (
        word_embedding,
        word_ids,
        cumsum,
        position_mask,
        position_embedding,
        global_embedding,
        weight,
        bias,
    ) = _validate_inputs(inputs)

    out = torch.empty_strided(
        (BATCH, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=word_embedding.device,
        dtype=torch.float16,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["XBLOCK"]),)
    _longformer_embedding_position_layernorm_kernel[grid](
        word_embedding,
        word_ids,
        cumsum,
        position_mask,
        position_embedding,
        global_embedding,
        weight,
        bias,
        out,
        hidden=HIDDEN,
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
