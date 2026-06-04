"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DistilBERT embedding assembly and layernorm return as a hand-specialized Triton row kernel, including token embedding, position embedding, fp16 embedding-add rounding, fp32 variance/mean over hidden size 768 with eps=1e-12, affine epilogue, fp16 cast, and all three [512, 768] view outputs, whereas Inductor currently emits one generic persistent-reduction Triton kernel that fuses the embedding loads, add, var_mean, affine, fp16 store, and aliasing view returns but carries generic reduction tiling, index assertions, and template bookkeeping; Inductor cannot emit this exact hand-specialized form today because its scheduler/codegen path recognizes only the decomposed reduction fusion, not a semantic DistilBERT embedding-layernorm pattern with fixed hidden size, direct indexed loads, and sibling view returns; the fix is NEW_PATTERN: add an embedding-layernorm lowering that folds indexed embedding loads into a fixed-hidden row layernorm kernel and preserves sibling view outputs over the same result buffer."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


SEQ_LEN = 512
HIDDEN = 768
BLOCK_H = 1024
VOCAB = 30522
POSITION_VOCAB = 512
EPS = 1.0e-12
DEFAULT_ROW_BLOCK = 1
DEFAULT_NUM_WARPS = 4
CLASSIFICATION = "NEW_PATTERN"


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _embedding_layernorm_kernel(
        word_table_ptr,
        token_ids_ptr,
        position_table_ptr,
        position_ids_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        eps: tl.constexpr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
    ):
        rows = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = rows < seq_len
        mask = cols < hidden

        token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)
        position_id = tl.load(position_ids_ptr + rows, mask=row_mask, other=0)

        token = tl.load(
            word_table_ptr + token_id * hidden + cols,
            mask=mask & row_mask,
            other=0.0,
        )
        position = tl.load(
            position_table_ptr + position_id * hidden + cols,
            mask=mask & row_mask,
            other=0.0,
        )
        x = (token + position).to(tl.float16).to(tl.float32)
        full_mask = mask & row_mask
        x_masked = tl.where(full_mask, x, 0.0)

        mean = tl.sum(x_masked, axis=1)[:, None] / hidden
        sum_x2 = tl.sum(x_masked * x_masked, axis=1)[:, None]
        var = sum_x2 / hidden - mean * mean
        invstd = tl.rsqrt(tl.maximum(var, 0.0) + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * invstd * weight + bias

        tl.store(out_ptr + rows * hidden + cols, y.to(tl.float16), mask=full_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 9:
        raise ValueError(f"expected 9 inputs, got {len(inputs)}")

    (
        word_table,
        token_ids,
        position_table,
        position_ids,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
    ) = inputs

    tensor_inputs = (
        word_table,
        token_ids,
        position_table,
        position_ids,
        weight,
        bias,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first six repro inputs must be tensors")

    expected_shapes = (
        (VOCAB, HIDDEN),
        (1, SEQ_LEN),
        (POSITION_VOCAB, HIDDEN),
        (1, SEQ_LEN),
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

    fp16_tensors = (word_table, position_table, weight, bias)
    if not all(value.dtype == torch.float16 for value in fp16_tensors):
        raise TypeError("embedding tables, layernorm weight, and bias must be torch.float16")
    if token_ids.dtype != torch.int64 or position_ids.dtype != torch.int64:
        raise TypeError("token and position ids must be torch.int64")

    output_shapes = (_shape_tuple(shape0), _shape_tuple(shape1), _shape_tuple(shape2))
    for index, shape in enumerate(output_shapes):
        if shape != (SEQ_LEN, HIDDEN):
            raise ValueError(f"shape parameter {index} is {shape}, expected {(SEQ_LEN, HIDDEN)}")

    return (
        word_table,
        token_ids,
        position_table,
        position_ids,
        weight,
        bias,
        *output_shapes,
    )


def oracle_embedding_layernorm(
    word_table: torch.Tensor,
    token_ids: torch.Tensor,
    position_table: torch.Tensor,
    position_ids: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    row_block: int = DEFAULT_ROW_BLOCK,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """Compute the pre-view fp16 [1, 512, 768] DistilBERT embedding layernorm output."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    out = torch.empty_strided(
        (1, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=word_table.device,
        dtype=torch.float16,
    )
    _embedding_layernorm_kernel[(triton.cdiv(SEQ_LEN, row_block),)](
        word_table,
        token_ids,
        position_table,
        position_ids,
        weight,
        bias,
        out,
        eps=EPS,
        seq_len=SEQ_LEN,
        hidden=HIDDEN,
        block_h=BLOCK_H,
        row_block=row_block,
        num_warps=num_warps,
    )
    return out


def oracle_forward(inputs):
    """Run the full Repro.forward computation and return all three view outputs."""
    (
        word_table,
        token_ids,
        position_table,
        position_ids,
        weight,
        bias,
        _shape0,
        _shape1,
        _shape2,
    ) = inputs

    out = oracle_embedding_layernorm(
        word_table,
        token_ids,
        position_table,
        position_ids,
        weight,
        bias,
    )
    return (
        out.view(SEQ_LEN, HIDDEN),
        out.view(SEQ_LEN, HIDDEN),
        out.view(SEQ_LEN, HIDDEN),
    )


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
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
