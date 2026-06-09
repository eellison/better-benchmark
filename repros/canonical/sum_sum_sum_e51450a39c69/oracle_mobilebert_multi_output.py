"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete MobileBERT multi-output reduction scope with one row-tiled Triton producer plus a finalizer, writing the required `[128,32768]` scaled side-output view and accumulating the two independent `[128]` column sums from the same loads, then deriving the third `[128]` reduction as `arg25 * sum(mm_694)` instead of separately summing the scaled side tensor; Inductor currently treats `sum(mm_694 * arg25)` as a sibling reduction over a materialized pointwise producer and does not prove that the per-hidden scale can be pulled out of the row reduction; the fix is ALGEBRAIC_ELIMINATION: teach reduction simplification to recognize broadcasts constant along reduced dimensions and reuse sibling sums for dependent scaled reductions while preserving the required side-output layout."""
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

BATCH = 256
TOKENS = 128
HIDDEN = 128
ROWS = BATCH * TOKENS


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"xblock": 1, "rblock": 128}, num_warps=4, num_stages=1),
            triton.Config({"xblock": 2, "rblock": 128}, num_warps=4, num_stages=1),
            triton.Config({"xblock": 4, "rblock": 128}, num_warps=4, num_stages=1),
            triton.Config({"xblock": 8, "rblock": 128}, num_warps=8, num_stages=1),
            triton.Config({"xblock": 16, "rblock": 128}, num_warps=8, num_stages=1),
            triton.Config({"xblock": 32, "rblock": 128}, num_warps=8, num_stages=1),
        ],
        key=["xnumel"],
    )
    @triton.jit
    def _partials_and_side_kernel(
        mm_ptr,
        arg609_ptr,
        arg25_ptr,
        side_ptr,
        partial0_ptr,
        partial1_ptr,
        mm_s0: tl.constexpr,
        mm_s1: tl.constexpr,
        arg609_s0: tl.constexpr,
        arg609_s1: tl.constexpr,
        arg609_s2: tl.constexpr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        xnumel: tl.constexpr,
        xblock: tl.constexpr,
        rblock: tl.constexpr,
    ):
        xindex = tl.program_id(0) * xblock + tl.arange(0, xblock)[:, None]
        token_offsets = tl.arange(0, rblock)[None, :]

        hidden_index = xindex % hidden
        batch_index = xindex // hidden

        mm_offsets = (batch_index * tokens + token_offsets) * mm_s0 + hidden_index * mm_s1
        arg609_offsets = (
            batch_index * arg609_s0
            + token_offsets * arg609_s1
            + hidden_index * arg609_s2
        )
        values = tl.load(mm_ptr + mm_offsets).to(tl.float32)
        arg609 = tl.load(arg609_ptr + arg609_offsets).to(tl.float32)
        scale = tl.load(arg25_ptr + hidden_index).to(tl.float32)
        side = values * scale

        flat_offsets = (batch_index * tokens + token_offsets) * hidden + hidden_index
        tl.store(side_ptr + flat_offsets, side)

        partial_offsets = batch_index * hidden + hidden_index
        tl.store(partial0_ptr + partial_offsets, tl.sum(values, axis=1)[:, None])
        tl.store(partial1_ptr + partial_offsets, tl.sum(values * arg609, axis=1)[:, None])

    @triton.autotune(
        configs=[
            triton.Config({"block_batch": 256}, num_warps=1, num_stages=1),
            triton.Config({"block_batch": 256}, num_warps=4, num_stages=1),
            triton.Config({"block_batch": 256}, num_warps=8, num_stages=1),
        ],
        key=["batch"],
    )
    @triton.jit
    def _finalize_partials_kernel(
        partial0_ptr,
        partial1_ptr,
        arg25_ptr,
        out0_ptr,
        out1_ptr,
        out3_ptr,
        batch: tl.constexpr,
        hidden: tl.constexpr,
        block_batch: tl.constexpr,
    ):
        col = tl.program_id(0)
        batch_offsets = tl.arange(0, block_batch)
        partial_offsets = batch_offsets * hidden + col
        acc0 = tl.load(partial0_ptr + partial_offsets)
        acc1 = tl.load(partial1_ptr + partial_offsets)
        sum0 = tl.sum(acc0, axis=0)
        sum1 = tl.sum(acc1, axis=0)
        scale = tl.load(arg25_ptr + col).to(tl.float32)

        tl.store(out0_ptr + col, sum0)
        tl.store(out1_ptr + col, sum1)
        tl.store(out3_ptr + col, sum0 * scale)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        mm_694,
        arg609_1,
        arg25_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs

    tensors = (mm_694, arg609_1, arg25_1)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("first three inputs must be tensors")

    expected = (
        ((ROWS, HIDDEN), torch.float32),
        ((BATCH, TOKENS, HIDDEN), torch.float32),
        ((HIDDEN,), torch.float32),
    )
    for index, (tensor, (shape, dtype)) in enumerate(zip(tensors, expected)):
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"input {index} expected shape={shape} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )

    expected_shapes = (
        (BATCH, TOKENS, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
        (ROWS, HIDDEN),
        (HIDDEN,),
    )
    actual_shapes = tuple(
        _shape_tuple(value) for value in (shape0, shape1, shape2, shape3, shape4)
    )
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"shape parameters do not match repro: actual={actual_shapes}, "
            f"expected={expected_shapes}"
        )

    return mm_694, arg609_1, arg25_1


def _torch_full_scope(inputs):
    (
        mm_694,
        arg609_1,
        arg25_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs
    view_default = torch.ops.aten.view.default(mm_694, shape0)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default, [0, 1], True)
    view_default_1 = torch.ops.aten.view.default(sum_dim_int_list, shape1)
    mul_tensor = torch.ops.aten.mul.Tensor(view_default, arg609_1)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default, arg25_1)
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True)
    view_default_2 = torch.ops.aten.view.default(sum_dim_int_list_1, shape2)
    view_default_3 = torch.ops.aten.view.default(mul_tensor_1, shape3)
    permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0])
    sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True)
    view_default_4 = torch.ops.aten.view.default(sum_dim_int_list_2, shape4)
    return view_default_1, view_default_2, permute_default, view_default_4


@oracle_impl(hardware="H100", shapes="(T([32768, 128], f32), T([256, 128, 128], f32), T([128], f32), S([256, 128, 128]), S([128]), S([128]), S([32768, 128]), S([128]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope with the same output count/layouts."""
    mm_694, arg609_1, arg25_1 = _validate_inputs(inputs)
    if triton is None or mm_694.device.type != "cuda":
        return _torch_full_scope(inputs)

    side_base = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=mm_694.device,
        dtype=torch.float32,
    )
    partial0 = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=mm_694.device,
        dtype=torch.float32,
    )
    partial1 = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=mm_694.device,
        dtype=torch.float32,
    )
    out0 = torch.empty_strided((HIDDEN,), (1,), device=mm_694.device, dtype=torch.float32)
    out1 = torch.empty_strided((HIDDEN,), (1,), device=mm_694.device, dtype=torch.float32)
    out3 = torch.empty_strided((HIDDEN,), (1,), device=mm_694.device, dtype=torch.float32)

    xnumel = BATCH * HIDDEN
    token_grid = lambda meta: (triton.cdiv(xnumel, meta["xblock"]),)
    _partials_and_side_kernel[token_grid](
        mm_694,
        arg609_1,
        arg25_1,
        side_base,
        partial0,
        partial1,
        mm_s0=mm_694.stride(0),
        mm_s1=mm_694.stride(1),
        arg609_s0=arg609_1.stride(0),
        arg609_s1=arg609_1.stride(1),
        arg609_s2=arg609_1.stride(2),
        tokens=TOKENS,
        hidden=HIDDEN,
        xnumel=xnumel,
    )
    _finalize_partials_kernel[(HIDDEN,)](
        partial0,
        partial1,
        arg25_1,
        out0,
        out1,
        out3,
        batch=BATCH,
        hidden=HIDDEN,
    )
    return out0, out1, side_base.permute(1, 0), out3


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
