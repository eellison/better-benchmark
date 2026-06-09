"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BART token embedding, generated position embedding with offset 2, fp32 population var_mean over hidden size 1024, eps=1e-5 rsqrt normalization, affine scale/bias epilogue, and three sibling [8192, 1024] views from one output buffer in one shape-specialized Triton row kernel, whereas Inductor already measures at the same floor for this full decomposed embedding/iota/var_mean/affine/view graph; Inductor cannot materially improve this local repro by adding a semantic BART embedding-LayerNorm pattern because the required indexed embedding reads, row reduction, affine reads, rsqrt, and one shared output store dominate after the existing fusion; the fix is BANDWIDTH_BOUND: record this as an at-floor embedding LayerNorm case unless broader normalization-template or memory-traffic work moves both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None


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


BATCH = 8
SEQ_LEN = 1024
HIDDEN = 1024
ROWS = BATCH * SEQ_LEN
VOCAB = 50265
POSITION_ROWS = 1026
POSITION_OFFSET = 2
EPS = 1.0e-5
OUTPUT_SHAPE = (ROWS, HIDDEN)
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _bart_embedding_layernorm_kernel(
        token_table_ptr,
        token_ids_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        position_offset: tl.constexpr,
        eps: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
        cols = tl.arange(0, hidden)[None, :]

        token_id = tl.load(token_ids_ptr + rows)
        position_id = (rows % seq_len) + position_offset

        token = tl.load(
            token_table_ptr + token_id * hidden + cols,
            eviction_policy="evict_first",
        ).to(tl.float32)
        position = tl.load(
            position_table_ptr + position_id * hidden + cols,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = token * 1.0 + position

        mean = tl.sum(x, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(centered * centered, axis=1)[:, None] / hidden
        invstd = libdevice.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + rows * hidden + cols, y)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter, got {type(value).__name__}")


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= max(int(size), 1)
    return tuple(reversed(strides))


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
    ) = inputs
    tensor_inputs = (token_table, token_ids, position_table, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (
        (VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (POSITION_ROWS, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.device != token_table.device:
            raise ValueError(f"input {index} device {value.device} != {token_table.device}")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if token_table.dtype != torch.float32 or position_table.dtype != torch.float32:
        raise TypeError("embedding tables must be torch.float32")
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("LayerNorm weight and bias must be torch.float32")
    if token_ids.dtype != torch.int64:
        raise TypeError(f"token ids must be torch.int64, got {token_ids.dtype}")

    output_shapes = (_shape_tuple(shape0), _shape_tuple(shape1), _shape_tuple(shape2))
    for index, shape in enumerate(output_shapes):
        if shape != OUTPUT_SHAPE:
            raise ValueError(f"shape parameter {index} is {shape}, expected {OUTPUT_SHAPE}")

    return (
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        *output_shapes,
    )


def _torch_full_scope(inputs: tuple[Any, ...] | list[Any]):
    token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2 = _validate_inputs(inputs)
    token = torch.ops.aten.embedding.default(token_table, token_ids, 1)
    scaled = torch.ops.aten.mul.Tensor(token, 1.0)
    position_ids = torch.ops.aten.add.Tensor(
        torch.ops.aten.unsqueeze.default(
            torch.ops.aten.add.Tensor(
                torch.ops.prims.iota.default(
                    SEQ_LEN,
                    start=0,
                    step=1,
                    dtype=torch.int64,
                    device=token_ids.device,
                    requires_grad=False,
                ),
                0,
            ),
            0,
        ),
        POSITION_OFFSET,
    )
    position = torch.ops.aten.embedding.default(position_table, position_ids)
    x = torch.ops.aten.add.Tensor(scaled, position)
    variance, mean = torch.ops.aten.var_mean.correction(x, [2], correction=0, keepdim=True)
    centered = torch.ops.aten.sub.Tensor(x, mean)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(centered, invstd)
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, weight), bias)
    return (
        torch.ops.aten.view.default(affine, shape0),
        torch.ops.aten.view.default(affine, shape1),
        torch.ops.aten.view.default(affine, shape2),
    )


def oracle_forward(inputs):
    """Run the full Repro.forward computation and return all three view outputs."""
    token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2 = _validate_inputs(inputs)
    if (
        triton is None
        or libdevice is None
        or not token_table.is_cuda
        or not token_ids.is_cuda
        or not position_table.is_cuda
        or not weight.is_cuda
        or not bias.is_cuda
    ):
        return _torch_full_scope(inputs)

    out = torch.empty_strided(
        OUTPUT_SHAPE,
        _contiguous_strides(OUTPUT_SHAPE),
        device=token_table.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _bart_embedding_layernorm_kernel[grid](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        position_offset=POSITION_OFFSET,
        eps=EPS,
    )
    return (
        torch.ops.aten.view.default(out, shape0),
        torch.ops.aten.view.default(out, shape1),
        torch.ops.aten.view.default(out, shape2),
    )


def _storage_id(value: torch.Tensor) -> int:
    return value.untyped_storage().data_ptr()


def _layout_alias_check(instance, inputs) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)

    eager_list = tuple(eager)
    oracle_list = tuple(oracle_out)
    ok = True
    if len(eager_list) != len(oracle_list):
        print(f"  layout/alias: FAIL output-count oracle={len(oracle_list)} eager={len(eager_list)}")
        return False

    for index, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        layout_ok = (
            expected.shape == actual.shape
            and expected.dtype == actual.dtype
            and expected.stride() == actual.stride()
            and expected.storage_offset() == actual.storage_offset()
        )
        print(
            f"  layout output {index}: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual.shape)} dtype={actual.dtype} stride={actual.stride()} "
            f"storage_offset={actual.storage_offset()})"
        )
        ok = ok and layout_ok

    expected_aliases = [
        [_storage_id(lhs) == _storage_id(rhs) for rhs in eager_list]
        for lhs in eager_list
    ]
    actual_aliases = [
        [_storage_id(lhs) == _storage_id(rhs) for rhs in oracle_list]
        for lhs in oracle_list
    ]
    alias_ok = expected_aliases == actual_aliases
    print(f"  output aliasing: {'PASS' if alias_ok else 'FAIL'}")
    return ok and alias_ok


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
        layout_ok = _layout_alias_check(instance, inputs)
        ok = ok and layout_ok
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
