"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full Bart inference embedding-layernorm graph as one shape-specialized Triton row kernel, including token embedding, generated position iota/add embedding, fp16 embedding-add rounding, fp32 var_mean layernorm with eps=1e-5, affine, fp16 cast, and the three returned [512, 768] views, whereas Inductor currently lowers the embedding gathers/iota arithmetic and generic var_mean layernorm through separate/general scheduler work with extra materialized intermediates and generic masked-reduction bookkeeping; Inductor cannot do this today because norm-template canonicalization does not recognize embedding plus affine-position producers as a single semantic embedding-layernorm pattern with direct indexed loads and multi-view output aliasing; the fix is NEW_PATTERN: add a Bart-style embedding-layernorm template that folds generated position ids and token/position table loads into a fixed-hidden row layernorm kernel and returns all view aliases from the one fp16 result."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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
BLOCK_N = 1024
TOKEN_VOCAB = 50265
POSITION_ROWS = 1026
POSITION_OFFSET = 2
EPS = 1.0e-5
CLASSIFICATION = "NEW_PATTERN"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _bart_embedding_layernorm_kernel(
        token_table_ptr,
        token_ids_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
        position_offset: tl.constexpr,
        eps: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_n)
        mask = cols < hidden

        token_id = tl.load(token_ids_ptr + row)
        token = tl.load(token_table_ptr + token_id * hidden + cols, mask=mask, other=0.0)
        position = tl.load(
            position_table_ptr + (row + position_offset) * hidden + cols,
            mask=mask,
            other=0.0,
        )

        summed = (token + position).to(tl.float16).to(tl.float32)
        sum_x = tl.sum(summed, axis=0)
        sum_x2 = tl.sum(summed * summed, axis=0)
        mean = sum_x / hidden
        var = sum_x2 / hidden - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (summed - mean) * invstd * weight + bias

        tl.store(out_ptr + row * hidden + cols, y.to(tl.float16), mask=mask)


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, Any, Any]:
    if len(inputs) != 8:
        raise ValueError(f"expected 8 inputs, got {len(inputs)}")

    token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2 = inputs
    tensor_inputs = (token_table, token_ids, position_table, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (
        (TOKEN_VOCAB, HIDDEN),
        (1, SEQ_LEN),
        (POSITION_ROWS, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")

    if token_table.dtype != torch.float16 or position_table.dtype != torch.float16:
        raise TypeError("embedding tables must be torch.float16")
    if weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise TypeError("layernorm weight and bias must be torch.float16")
    if token_ids.dtype != torch.int64:
        raise TypeError("token ids must be torch.int64")

    for index, shape in enumerate((shape0, shape1, shape2), start=5):
        if list(shape) != [SEQ_LEN, HIDDEN]:
            raise ValueError(f"input {index} unexpected shape parameter: {shape!r}")

    return token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2


def oracle_embedding_layernorm(
    token_table: torch.Tensor,
    token_ids: torch.Tensor,
    position_table: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    shape0: Any,
    shape1: Any,
    shape2: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the complete Bart embedding + layernorm graph and return all views."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    out = torch.empty_strided(
        (SEQ_LEN, HIDDEN),
        (HIDDEN, 1),
        device=token_table.device,
        dtype=torch.float16,
    )
    _bart_embedding_layernorm_kernel[(SEQ_LEN,)](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        block_n=BLOCK_N,
        position_offset=POSITION_OFFSET,
        eps=EPS,
        num_warps=1,
        num_stages=4,
    )
    return out, out, out


@oracle_impl(hardware="H100", shapes="(T([50265, 768], f16), T([1, 512], i64, gen=Index(50265)), T([1026, 768], f16), T([768], f16), T([768], f16), S([512, 768]), S([512, 768]), S([512, 768]))")
def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full-scope oracle for Repro.forward."""
    return oracle_embedding_layernorm(*_validate_inputs(inputs))


def main():
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
