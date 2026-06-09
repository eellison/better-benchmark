"""
Oracle for var_mean_a4b0facb2977

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete
RoBERTa embedding assembly and LayerNorm in one Triton row kernel, including the
word embedding lookup, cumsum/mask-derived position ids, token-type gather,
token-type and position embedding lookups, fp16 embedding-add rounding, fp32
`var_mean(..., dim=2, correction=0, keepdim=True)`, affine epilogue, fp16 cast,
and the three final `[512, 768]` views, whereas Inductor emits a single generic
fused reduction kernel for the decomposed embedding/gather/add/LayerNorm graph;
Inductor cannot emit this exact floor today because norm-template
canonicalization does not recognize this embedding-layernorm shape as a
dedicated direct-indexed-load pattern with fixed hidden size, fp16 intermediate
rounding, and multi-view output aliasing; the fix is NEW_PATTERN: add an
embedding-layernorm template that folds word/type/position indexed loads into a
fixed-K row-normalization kernel and autotunes this small hidden-size family.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

SEQ_LEN = 512
HIDDEN = 768
WORD_VOCAB = 250002
POSITION_VOCAB = 514
TOKEN_TYPE_VOCAB = 1
EPS = 1.0e-5
BLOCK_H = 1024
DEFAULT_NUM_WARPS = 1
CLASSIFICATION = "NEW_PATTERN"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _embedding_layernorm_kernel(
        word_table_ptr,
        token_ids_ptr,
        token_type_source_ptr,
        cumsum_ptr,
        position_mask_ptr,
        token_type_table_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        col_mask = cols < hidden

        token_id = tl.load(token_ids_ptr + row)
        cumsum_i32 = tl.load(cumsum_ptr + row).to(tl.int32)
        position_mask = tl.load(position_mask_ptr + row)
        position_id = (cumsum_i32 * position_mask).to(tl.int64) + 1
        token_type_id = tl.load(token_type_source_ptr + position_id)

        word = tl.load(
            word_table_ptr + token_id * hidden + cols,
            mask=col_mask,
            other=0.0,
        )
        token_type = tl.load(
            token_type_table_ptr + token_type_id * hidden + cols,
            mask=col_mask,
            other=0.0,
        )
        position = tl.load(
            position_table_ptr + position_id * hidden + cols,
            mask=col_mask,
            other=0.0,
        )

        # The captured graph rounds after each fp16 embedding add before
        # promoting to fp32 for var_mean/LayerNorm.
        pair_sum = (word + token_type).to(tl.float16)
        summed = (pair_sum + position).to(tl.float16).to(tl.float32)
        summed_for_reduce = tl.where(col_mask, summed, 0.0)

        mean = tl.sum(summed_for_reduce, axis=0) / hidden
        centered = summed - mean
        centered_for_var = tl.where(col_mask, centered, 0.0)
        variance = tl.sum(centered_for_var * centered_for_var, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        out = centered * invstd * weight + bias

        tl.store(out_ptr + row * hidden + cols, out.to(tl.float16), mask=col_mask)


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
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 13:
        raise ValueError(f"{REPRO_ID} expects 13 inputs, got {len(inputs)}")

    (
        word_table,
        token_ids,
        token_type_source,
        cumsum,
        position_mask,
        token_type_table,
        position_table,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
        shape3,
    ) = inputs

    tensor_inputs = (
        word_table,
        token_ids,
        token_type_source,
        cumsum,
        position_mask,
        token_type_table,
        position_table,
        weight,
        bias,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first nine repro inputs must be tensors")

    expected_shapes = (
        (WORD_VOCAB, HIDDEN),
        (1, SEQ_LEN),
        (1, POSITION_VOCAB),
        (1, SEQ_LEN),
        (1, SEQ_LEN),
        (TOKEN_TYPE_VOCAB, HIDDEN),
        (POSITION_VOCAB, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")

    fp16_tensors = (word_table, token_type_table, position_table, weight, bias)
    if not all(value.dtype == torch.float16 for value in fp16_tensors):
        raise TypeError("embedding tables, layernorm weight, and bias must be torch.float16")

    int64_tensors = (token_ids, token_type_source, cumsum)
    if not all(value.dtype == torch.int64 for value in int64_tensors):
        raise TypeError("token ids, token-type source, and cumsum must be torch.int64")
    if position_mask.dtype != torch.int32:
        raise TypeError(f"position mask must be torch.int32, got {position_mask.dtype}")

    shape0_tuple = tuple(int(dim) for dim in shape0)
    shape1_tuple = tuple(int(dim) for dim in shape1)
    shape2_tuple = tuple(int(dim) for dim in shape2)
    shape3_tuple = tuple(int(dim) for dim in shape3)
    if shape0_tuple != (1, SEQ_LEN):
        raise ValueError(f"unexpected gather expand shape parameter: {shape0!r}")
    for index, shape_tuple in enumerate((shape1_tuple, shape2_tuple, shape3_tuple), start=1):
        if shape_tuple != (SEQ_LEN, HIDDEN):
            raise ValueError(f"unexpected output shape{index} parameter: {shape_tuple!r}")

    return (
        word_table,
        token_ids,
        token_type_source,
        cumsum,
        position_mask,
        token_type_table,
        position_table,
        weight,
        bias,
        shape0_tuple,
        shape1_tuple,
        shape2_tuple,
        shape3_tuple,
    )


def oracle_embedding_layernorm(
    word_table: torch.Tensor,
    token_ids: torch.Tensor,
    token_type_source: torch.Tensor,
    cumsum: torch.Tensor,
    position_mask: torch.Tensor,
    token_type_table: torch.Tensor,
    position_table: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    shape1: tuple[int, int],
    shape2: tuple[int, int],
    shape3: tuple[int, int],
    *,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the full Repro.forward output with one Triton row kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_embedding_layernorm.py")

    base = torch.empty_strided(
        (1, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=word_table.device,
        dtype=torch.float16,
    )
    _embedding_layernorm_kernel[(SEQ_LEN,)](
        word_table,
        token_ids,
        token_type_source,
        cumsum,
        position_mask,
        token_type_table,
        position_table,
        weight,
        bias,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )

    return base.view(shape1), base.view(shape2), base.view(shape3)


@oracle_impl(hardware="H100", shapes="(T([250002, 768], f16), T([1, 512], i64, gen=Index(250002)), T([1, 514], i64, gen=Index(1)), T([1, 512], i64, gen=Index(513)), T([1, 512], i32, gen=Index(2)), T([1, 768], f16), T([514, 768], f16), T([768], f16), T([768], f16), S([1, 512]), S([512, 768]), S([512, 768]), S([512, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the complete Repro.forward embedding + LayerNorm computation.

    SCOPE INVARIANT: accepts the same 13 inputs as Repro.forward() and returns
    the same three f16 `[512, 768]` outputs with contiguous `(768, 1)` stride.
    """
    (
        word_table,
        token_ids,
        token_type_source,
        cumsum,
        position_mask,
        token_type_table,
        position_table,
        weight,
        bias,
        _shape0,
        shape1,
        shape2,
        shape3,
    ) = _validate_inputs(inputs)
    return oracle_embedding_layernorm(
        word_table,
        token_ids,
        token_type_source,
        cumsum,
        position_mask,
        token_type_table,
        position_table,
        weight,
        bias,
        shape1,
        shape2,
        shape3,
    )


def _check_layout_and_alias(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected, actual)):
        layout_ok = (
            tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
            and actual_tensor.stride() == expected_tensor.stride()
            and actual_tensor.dtype == expected_tensor.dtype
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual_tensor.shape)} stride={actual_tensor.stride()})"
        )
        ok = ok and layout_ok

    expected_alias = len({tensor.data_ptr() for tensor in expected}) == 1
    actual_alias = len({tensor.data_ptr() for tensor in actual}) == 1
    alias_ok = expected_alias == actual_alias
    print(
        f"  output aliasing: {'PASS' if alias_ok else 'FAIL'} "
        f"(expected_same_ptr={expected_alias} oracle_same_ptr={actual_alias})"
    )
    return ok and alias_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
        ok = _check_layout_and_alias(instance, inputs) and ok
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
