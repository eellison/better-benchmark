"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete MobileBERT eight-output reduction scope with one row-tiled Triton producer that writes both required `[128, 32768]` scaled side-output views and accumulates all sibling/dependent `[128]` reductions from the same loaded values, whereas Inductor schedules the captured add/mul producers, duplicate scaled reductions, and two side-output transposes as generic multi-output reductions over materialized intermediates; Inductor cannot do this today because its algebraic reduction simplifier does not recognize broadcast scales and duplicate view reductions as reusable summaries while preserving returned side views; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to reuse multi-output column summaries across broadcast-scaled dependent reductions and emit the side-view writers from the shared producer."""
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


BATCH = 256
TOKENS = 128
HIDDEN = 128
ROWS = BATCH * TOKENS

INPUT_2D_SHAPE = (ROWS, HIDDEN)
INPUT_2D_STRIDE = (HIDDEN, 1)
INPUT_3D_SHAPE = (BATCH, TOKENS, HIDDEN)
INPUT_3D_STRIDE = (TOKENS * HIDDEN, HIDDEN, 1)
VECTOR_SHAPE = (HIDDEN,)
VECTOR_STRIDE = (1,)
SIDE_BASE_SHAPE = (ROWS, HIDDEN)
SIDE_BASE_STRIDE = (HIDDEN, 1)
VECTOR_PARTIAL_SHAPE = (BATCH, HIDDEN)
VECTOR_PARTIAL_STRIDE = (HIDDEN, 1)


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
        ],
        key=["xnumel"],
    )
    @triton.jit
    def _partials_and_side_kernel(
        mm_ptr,
        mul_ptr,
        arg588_ptr,
        arg5_ptr,
        arg6_ptr,
        arg597_ptr,
        arg13_ptr,
        side0_ptr,
        side1_ptr,
        partial0_ptr,
        partial1_ptr,
        partial3_ptr,
        partial5_ptr,
        partial7_ptr,
        mm_s0: tl.constexpr,
        mm_s1: tl.constexpr,
        mul_s0: tl.constexpr,
        mul_s1: tl.constexpr,
        mul_s2: tl.constexpr,
        arg588_s0: tl.constexpr,
        arg588_s1: tl.constexpr,
        arg597_s0: tl.constexpr,
        arg597_s1: tl.constexpr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        xnumel: tl.constexpr,
        xblock: tl.constexpr,
        rblock: tl.constexpr,
    ):
        xindex = tl.program_id(0) * xblock + tl.arange(0, xblock)[:, None]
        token_offsets = tl.arange(0, rblock)[None, :]
        xmask = xindex < xnumel

        hidden_index = xindex % hidden
        batch_index = xindex // hidden

        mm_offsets = (batch_index * tokens + token_offsets) * mm_s0 + hidden_index * mm_s1
        mul_offsets = (
            batch_index * mul_s0
            + token_offsets * mul_s1
            + hidden_index * mul_s2
        )
        arg588_offsets = (batch_index * tokens + token_offsets) * arg588_s0 + hidden_index * arg588_s1
        arg597_offsets = (batch_index * tokens + token_offsets) * arg597_s0 + hidden_index * arg597_s1

        mm_value = tl.load(mm_ptr + mm_offsets, mask=xmask, other=0.0).to(tl.float32)
        mul_value = tl.load(mul_ptr + mul_offsets, mask=xmask, other=0.0).to(tl.float32)
        arg588_value = tl.load(arg588_ptr + arg588_offsets, mask=xmask, other=0.0).to(tl.float32)
        arg597_value = tl.load(arg597_ptr + arg597_offsets, mask=xmask, other=0.0).to(tl.float32)
        scale5 = tl.load(arg5_ptr + hidden_index, mask=xmask, other=0.0).to(tl.float32)
        bias6 = tl.load(arg6_ptr + hidden_index, mask=xmask, other=0.0).to(tl.float32)
        scale13 = tl.load(arg13_ptr + hidden_index, mask=xmask, other=0.0).to(tl.float32)

        add_tensor = mul_value + mm_value
        mul_tensor = arg588_value * scale5
        add_tensor_1 = mul_tensor + bias6
        add_tensor_2 = arg597_value + add_tensor_1
        mul_tensor_1 = add_tensor * add_tensor_2
        mul_tensor_2 = add_tensor * scale13
        mul_tensor_3 = mul_tensor_2 * arg588_value
        mul_tensor_4 = mul_tensor_2 * scale5

        flat_offsets = (batch_index * tokens + token_offsets) * hidden + hidden_index
        tl.store(side0_ptr + flat_offsets, mul_tensor_2, mask=xmask)
        tl.store(side1_ptr + flat_offsets, mul_tensor_4, mask=xmask)

        partial_offsets = batch_index * hidden + hidden_index
        tl.store(partial0_ptr + partial_offsets, tl.sum(add_tensor, axis=1)[:, None], mask=xmask)
        tl.store(partial1_ptr + partial_offsets, tl.sum(mul_tensor_1, axis=1)[:, None], mask=xmask)
        tl.store(partial3_ptr + partial_offsets, tl.sum(mul_tensor_2, axis=1)[:, None], mask=xmask)
        tl.store(partial5_ptr + partial_offsets, tl.sum(mul_tensor_3, axis=1)[:, None], mask=xmask)
        tl.store(partial7_ptr + partial_offsets, tl.sum(mul_tensor_4, axis=1)[:, None], mask=xmask)

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
        partial3_ptr,
        partial5_ptr,
        partial7_ptr,
        out0_ptr,
        out1_ptr,
        out3_ptr,
        out4_ptr,
        out5_ptr,
        out7_ptr,
        batch: tl.constexpr,
        hidden: tl.constexpr,
        block_batch: tl.constexpr,
    ):
        col = tl.program_id(0)
        batch_offsets = tl.arange(0, block_batch)
        mask = batch_offsets < batch
        partial_offsets = batch_offsets * hidden + col

        sum0 = tl.sum(tl.load(partial0_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        sum1 = tl.sum(tl.load(partial1_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        sum3 = tl.sum(tl.load(partial3_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        sum5 = tl.sum(tl.load(partial5_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        sum7 = tl.sum(tl.load(partial7_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32), axis=0)

        tl.store(out0_ptr + col, sum0)
        tl.store(out1_ptr + col, sum1)
        tl.store(out3_ptr + col, sum3)
        tl.store(out4_ptr + col, sum3)
        tl.store(out5_ptr + col, sum5)
        tl.store(out7_ptr + col, sum7)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 18:
        raise ValueError(f"{REPRO_ID} expects 18 inputs, got {len(inputs)}")

    (
        mm_710,
        mul_524,
        arg588_1,
        arg5_1,
        arg6_1,
        arg597_1,
        arg13_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
        shape7,
        shape8,
        shape9,
        shape10,
    ) = inputs

    tensors = (
        _expect_tensor("mm_710", mm_710, INPUT_2D_SHAPE, INPUT_2D_STRIDE),
        _expect_tensor("mul_524", mul_524, INPUT_3D_SHAPE, INPUT_3D_STRIDE),
        _expect_tensor("arg588_1", arg588_1, INPUT_2D_SHAPE, INPUT_2D_STRIDE),
        _expect_tensor("arg5_1", arg5_1, VECTOR_SHAPE, VECTOR_STRIDE),
        _expect_tensor("arg6_1", arg6_1, VECTOR_SHAPE, VECTOR_STRIDE),
        _expect_tensor("arg597_1", arg597_1, INPUT_2D_SHAPE, INPUT_2D_STRIDE),
        _expect_tensor("arg13_1", arg13_1, VECTOR_SHAPE, VECTOR_STRIDE),
    )

    expected_shapes = (
        INPUT_3D_SHAPE,
        VECTOR_SHAPE,
        INPUT_3D_SHAPE,
        INPUT_3D_SHAPE,
        VECTOR_SHAPE,
        INPUT_2D_SHAPE,
        VECTOR_SHAPE,
        VECTOR_SHAPE,
        VECTOR_SHAPE,
        INPUT_2D_SHAPE,
        VECTOR_SHAPE,
    )
    actual_shapes = tuple(
        _shape_tuple(value)
        for value in (
            shape0,
            shape1,
            shape2,
            shape3,
            shape4,
            shape5,
            shape6,
            shape7,
            shape8,
            shape9,
            shape10,
        )
    )
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"shape parameters do not match repro: actual={actual_shapes}, "
            f"expected={expected_shapes}"
        )

    device = tensors[0].device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same device")
    return tensors


def _torch_full_scope(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    (
        mm_710,
        mul_524,
        arg588_1,
        arg5_1,
        arg6_1,
        arg597_1,
        arg13_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
        shape7,
        shape8,
        shape9,
        shape10,
    ) = inputs
    view_default = torch.ops.aten.view.default(mm_710, shape0)
    add_tensor = torch.ops.aten.add.Tensor(mul_524, view_default)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1], True)
    view_default_1 = torch.ops.aten.view.default(sum_dim_int_list, shape1)
    view_default_2 = torch.ops.aten.view.default(arg588_1, shape2)
    mul_tensor = torch.ops.aten.mul.Tensor(view_default_2, arg5_1)
    add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor, arg6_1)
    view_default_3 = torch.ops.aten.view.default(arg597_1, shape3)
    add_tensor_2 = torch.ops.aten.add.Tensor(view_default_3, add_tensor_1)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor, add_tensor_2)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor, arg13_1)
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True)
    view_default_4 = torch.ops.aten.view.default(sum_dim_int_list_1, shape4)
    view_default_5 = torch.ops.aten.view.default(mul_tensor_2, shape5)
    permute_default = torch.ops.aten.permute.default(view_default_5, [1, 0])
    sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_5, [0], True)
    view_default_6 = torch.ops.aten.view.default(sum_dim_int_list_2, shape6)
    sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True)
    view_default_7 = torch.ops.aten.view.default(sum_dim_int_list_3, shape7)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, view_default_2)
    mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, arg5_1)
    sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True)
    view_default_8 = torch.ops.aten.view.default(sum_dim_int_list_4, shape8)
    view_default_9 = torch.ops.aten.view.default(mul_tensor_4, shape9)
    permute_default_1 = torch.ops.aten.permute.default(view_default_9, [1, 0])
    sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(view_default_9, [0], True)
    view_default_10 = torch.ops.aten.view.default(sum_dim_int_list_5, shape10)
    return (
        view_default_1,
        view_default_4,
        permute_default,
        view_default_6,
        view_default_7,
        view_default_8,
        permute_default_1,
        view_default_10,
    )


@oracle_impl(hardware="H100", shapes="(T([32768, 128], f32), T([256, 128, 128], f32), T([32768, 128], f32), T([128], f32), T([128], f32), T([32768, 128], f32), T([128], f32), S([256, 128, 128]), S([128]), S([256, 128, 128]), S([256, 128, 128]), S([128]), S([32768, 128]), S([128]), S([128]), S([128]), S([32768, 128]), S([128]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope with the same output count/layouts."""
    (
        mm_710,
        mul_524,
        arg588_1,
        arg5_1,
        arg6_1,
        arg597_1,
        arg13_1,
    ) = _validate_inputs(inputs)
    if triton is None or mm_710.device.type != "cuda":
        return _torch_full_scope(inputs)

    side0_base = torch.empty_strided(
        SIDE_BASE_SHAPE,
        SIDE_BASE_STRIDE,
        device=mm_710.device,
        dtype=torch.float32,
    )
    side1_base = torch.empty_strided(
        SIDE_BASE_SHAPE,
        SIDE_BASE_STRIDE,
        device=mm_710.device,
        dtype=torch.float32,
    )
    partial0 = torch.empty_strided(
        VECTOR_PARTIAL_SHAPE,
        VECTOR_PARTIAL_STRIDE,
        device=mm_710.device,
        dtype=torch.float32,
    )
    partial1 = torch.empty_strided(
        VECTOR_PARTIAL_SHAPE,
        VECTOR_PARTIAL_STRIDE,
        device=mm_710.device,
        dtype=torch.float32,
    )
    partial3 = torch.empty_strided(
        VECTOR_PARTIAL_SHAPE,
        VECTOR_PARTIAL_STRIDE,
        device=mm_710.device,
        dtype=torch.float32,
    )
    partial5 = torch.empty_strided(
        VECTOR_PARTIAL_SHAPE,
        VECTOR_PARTIAL_STRIDE,
        device=mm_710.device,
        dtype=torch.float32,
    )
    partial7 = torch.empty_strided(
        VECTOR_PARTIAL_SHAPE,
        VECTOR_PARTIAL_STRIDE,
        device=mm_710.device,
        dtype=torch.float32,
    )
    out0 = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=mm_710.device, dtype=torch.float32)
    out1 = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=mm_710.device, dtype=torch.float32)
    out3 = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=mm_710.device, dtype=torch.float32)
    out4 = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=mm_710.device, dtype=torch.float32)
    out5 = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=mm_710.device, dtype=torch.float32)
    out7 = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=mm_710.device, dtype=torch.float32)

    xnumel = BATCH * HIDDEN
    token_grid = lambda meta: (triton.cdiv(xnumel, meta["xblock"]),)
    _partials_and_side_kernel[token_grid](
        mm_710,
        mul_524,
        arg588_1,
        arg5_1,
        arg6_1,
        arg597_1,
        arg13_1,
        side0_base,
        side1_base,
        partial0,
        partial1,
        partial3,
        partial5,
        partial7,
        mm_s0=mm_710.stride(0),
        mm_s1=mm_710.stride(1),
        mul_s0=mul_524.stride(0),
        mul_s1=mul_524.stride(1),
        mul_s2=mul_524.stride(2),
        arg588_s0=arg588_1.stride(0),
        arg588_s1=arg588_1.stride(1),
        arg597_s0=arg597_1.stride(0),
        arg597_s1=arg597_1.stride(1),
        tokens=TOKENS,
        hidden=HIDDEN,
        xnumel=xnumel,
    )
    _finalize_partials_kernel[(HIDDEN,)](
        partial0,
        partial1,
        partial3,
        partial5,
        partial7,
        out0,
        out1,
        out3,
        out4,
        out5,
        out7,
        batch=BATCH,
        hidden=HIDDEN,
    )
    return (
        out0,
        out1,
        side0_base.permute(1, 0),
        out3,
        out4,
        out5,
        side1_base.permute(1, 0),
        out7,
    )


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
