"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full GPT-J shifted-label cross-entropy backward tail in one Triton pass over all 128 rows for each vocabulary tile, storing the returned transposed gradient backing tensor and accumulating the returned vocabulary sum from the same computed values, while replacing the giant one-hot `where/eq/sum` with the equivalent per-row valid-label scale; whereas Inductor materializes the one-hot-shaped producer, reduces it to recover a row scalar, applies the exp epilogue, adds the residual tensor, then separately reduces the final tensor for the second output. Inductor cannot do this today because its reduction canonicalization does not prove that the one-hot masked row reduction is just `-arg629/arg600` for valid labels and zero otherwise, so the scheduler misses the full multi-output producer/reduction fusion. The fix is ALGEBRAIC_ELIMINATION: canonicalize one-hot masked reductions into label-indexed scalar formulas and sink the derived scalar into a fused multi-output reduction template that emits both returned outputs."""
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

ROWS = 128
VOCAB = 50400


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _full_backward_and_sum_kernel(
        numerator_ptr,
        denominator_ptr,
        labels_ptr,
        logits_ptr,
        row_shift0_ptr,
        row_shift1_ptr,
        residual_ptr,
        base_out_ptr,
        sum_out_ptr,
        rows_n: tl.constexpr,
        vocab_size: tl.constexpr,
        block_cols: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols_1d = pid * block_cols + tl.arange(0, block_cols)
        rows = tl.arange(0, rows_n)[:, None]
        cols = cols_1d[None, :]
        active = cols < vocab_size
        offsets = rows * vocab_size + cols

        scale_value = (
            tl.load(numerator_ptr).to(tl.float32)
            / tl.load(denominator_ptr).to(tl.float32)
        )
        labels = tl.load(labels_ptr + 1 + rows)
        valid = labels != -100
        safe_labels = tl.where(valid, labels, 0)
        scale = tl.where(valid, scale_value, 0.0)

        row_shift0 = tl.load(row_shift0_ptr + rows).to(tl.float32)
        row_shift1 = tl.load(row_shift1_ptr + rows).to(tl.float32)
        logits = tl.load(logits_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=active, other=0.0).to(tl.float32)

        exp_values = tl.exp(logits - row_shift0 - row_shift1)
        target_adjust = tl.where((safe_labels == cols) & valid, scale, 0.0)
        correction = scale * exp_values - target_adjust
        values = residual + correction

        tl.store(base_out_ptr + offsets, values, mask=active)
        col_sum = tl.sum(tl.where(active, values, 0.0), axis=0)
        tl.store(sum_out_ptr + cols_1d, col_sum, mask=cols_1d < vocab_size)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 13:
        raise ValueError(f"{REPRO_ID} expects 13 inputs, got {len(inputs)}")

    (
        arg629_1,
        arg600_1,
        arg597_1,
        arg596_1,
        arg598_1,
        arg599_1,
        arg630_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
    ) = inputs

    expected_tensors = (
        (arg629_1, (), torch.float32),
        (arg600_1, (), torch.float32),
        (arg597_1, (1, ROWS + 1), torch.int64),
        (arg596_1, (1, ROWS, VOCAB), torch.float32),
        (arg598_1, (ROWS, 1), torch.float32),
        (arg599_1, (ROWS, 1), torch.float32),
        (arg630_1, (1, ROWS, VOCAB), torch.float32),
    )
    for index, (tensor, shape, dtype) in enumerate(expected_tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"input {index} must be a tensor")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"input {index} expected shape={shape} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )

    expected_shapes = (
        (1, VOCAB),
        (ROWS, VOCAB),
        (-1, VOCAB),
        (1, ROWS, VOCAB),
        (ROWS, VOCAB),
        (VOCAB,),
    )
    actual_shapes = tuple(
        _shape_tuple(value) for value in (shape0, shape1, shape2, shape3, shape4, shape5)
    )
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"shape parameters do not match repro: actual={actual_shapes}, "
            f"expected={expected_shapes}"
        )

    return (
        arg629_1,
        arg600_1,
        arg597_1,
        arg596_1,
        arg598_1,
        arg599_1,
        arg630_1,
    )


def _torch_full_scope(inputs: tuple[Any, ...] | list[Any]):
    (
        arg629_1,
        arg600_1,
        arg597_1,
        arg596_1,
        arg598_1,
        arg599_1,
        arg630_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
    ) = inputs

    div_tensor = torch.ops.aten.div.Tensor(arg629_1, arg600_1)
    slice_tensor = torch.ops.aten.slice.Tensor(arg597_1, 1, 1, 9223372036854775807)
    view_default = torch.ops.aten.view.default(slice_tensor, [-1])
    unsqueeze_default = torch.ops.aten.unsqueeze.default(view_default, 1)
    ne_scalar = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
    full_default = torch.ops.aten.full.default(
        [],
        0,
        dtype=torch.int64,
        layout=torch.strided,
        device=arg597_1.device,
        pin_memory=False,
    )
    where_self = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default)
    iota_default = torch.ops.prims.iota.default(
        VOCAB,
        start=0,
        step=1,
        dtype=torch.int64,
        device=arg597_1.device,
        requires_grad=False,
    )
    view_default_1 = torch.ops.aten.view.default(iota_default, shape0)
    expand_default = torch.ops.aten.expand.default(where_self, shape1)
    eq_tensor = torch.ops.aten.eq.Tensor(expand_default, view_default_1)
    scalar_tensor_default = torch.ops.aten.scalar_tensor.default(
        0,
        dtype=torch.float32,
        layout=torch.strided,
        device=arg596_1.device,
    )
    scalar_tensor_default_1 = torch.ops.aten.scalar_tensor.default(
        -1.0,
        dtype=torch.float32,
        layout=torch.strided,
        device=arg596_1.device,
    )
    where_self_1 = torch.ops.aten.where.self(
        eq_tensor,
        scalar_tensor_default_1,
        scalar_tensor_default,
    )
    full_default_1 = torch.ops.aten.full.default(
        [],
        0.0,
        dtype=torch.float32,
        layout=torch.strided,
        device=arg596_1.device,
        pin_memory=False,
    )
    where_self_2 = torch.ops.aten.where.self(ne_scalar, div_tensor, full_default_1)
    mul_tensor = torch.ops.aten.mul.Tensor(where_self_1, where_self_2)
    view_default_2 = torch.ops.aten.view.default(arg596_1, shape2)
    sub_tensor = torch.ops.aten.sub.Tensor(view_default_2, arg598_1)
    sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, arg599_1)
    exp_default = torch.ops.aten.exp.default(sub_tensor_1)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list)
    sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1)
    view_default_3 = torch.ops.aten.view.default(sub_tensor_2, shape3)
    add_tensor = torch.ops.aten.add.Tensor(arg630_1, view_default_3)
    view_default_4 = torch.ops.aten.view.default(add_tensor, shape4)
    permute_default = torch.ops.aten.permute.default(view_default_4, [1, 0])
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_4, [0], True)
    view_default_5 = torch.ops.aten.view.default(sum_dim_int_list_1, shape5)
    return permute_default, view_default_5


def oracle_forward(inputs):
    """Run the full Repro.forward scope with the same output count and layouts."""
    (
        arg629_1,
        arg600_1,
        arg597_1,
        arg596_1,
        arg598_1,
        arg599_1,
        arg630_1,
    ) = _validate_inputs(inputs)

    if triton is None or arg596_1.device.type != "cuda":
        return _torch_full_scope(inputs)

    base_out = torch.empty_strided(
        (ROWS, VOCAB),
        (VOCAB, 1),
        device=arg596_1.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided(
        (VOCAB,),
        (1,),
        device=arg596_1.device,
        dtype=torch.float32,
    )

    block_cols = 64
    grid = (triton.cdiv(VOCAB, block_cols),)
    _full_backward_and_sum_kernel[grid](
        arg629_1,
        arg600_1,
        arg597_1,
        arg596_1,
        arg598_1,
        arg599_1,
        arg630_1,
        base_out,
        sum_out,
        rows_n=ROWS,
        vocab_size=VOCAB,
        block_cols=block_cols,
        num_warps=8,
    )
    return base_out.permute(1, 0), sum_out


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
