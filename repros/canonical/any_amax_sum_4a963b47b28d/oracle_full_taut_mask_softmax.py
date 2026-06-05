"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete captured M2M100 attention-softmax return from Repro.forward, including the [1024,128,128] to [64,16,128,128] score view, generated iota/add/unsqueeze/ge/expand mask, scalar where-to-0/-inf additive bias, row-all-minus-inf any guard, stable last-dimension softmax, zero fill for all-masked rows, expand, and final contiguous [1024,128,128] view, by proving the generated iota >= 0 predicate is tautologically true and folding the additive bias to zero before one Triton row-softmax kernel; Inductor currently lowers the decomposed view/iota/ge/where/add/eq/any/amax/sub/exp/sum/div/where/expand/view graph as generic pointwise and reduction work over materialized intermediates; Inductor cannot do this today because its scheduler/codegen simplifier does not prove shape-derived structured predicates through unsqueeze/expand/where before online-softmax scheduling; the fix is ALGEBRAIC_ELIMINATION: add shape-aware predicate and zero-bias simplification that canonicalizes tautological attention masks before reduction scheduling."""
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

CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _expand_param_matches(
    actual: tuple[int, ...],
    expected: tuple[int, ...],
) -> bool:
    return len(actual) == len(expected) and all(
        actual_dim == -1 or actual_dim == expected_dim
        for actual_dim, expected_dim in zip(actual, expected)
    )


def _validate_scope(
    bmm_24: torch.Tensor,
    shape_param_0: Any,
    shape_param_1: Any,
    shape_param_2: Any,
    shape_param_3: Any,
) -> tuple[tuple[int, int, int], int, int]:
    if not isinstance(bmm_24, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(bmm_24).__name__}")
    if bmm_24.dtype != torch.float32:
        raise TypeError(f"expected fp32 input, got {bmm_24.dtype}")
    if bmm_24.ndim != 3:
        raise ValueError(f"expected rank-3 input, got rank {bmm_24.ndim}")

    input_shape = _shape_tuple(bmm_24.shape)
    view_shape = _shape_tuple(shape_param_0)
    mask_expand_shape = _shape_tuple(shape_param_1)
    softmax_expand_shape = _shape_tuple(shape_param_2)
    output_shape = _shape_tuple(shape_param_3)

    if len(view_shape) != 4:
        raise ValueError(f"expected rank-4 first view shape, got {view_shape}")
    batch, heads, q_len, k_len = view_shape
    if (batch * heads, q_len, k_len) != input_shape:
        raise ValueError(
            f"view shape {view_shape} is not compatible with input shape {input_shape}"
        )
    expected_mask_expand = (batch, 1, q_len, k_len)
    if not _expand_param_matches(mask_expand_shape, expected_mask_expand):
        raise ValueError(
            f"mask expand shape {mask_expand_shape} does not match "
            f"{expected_mask_expand}"
        )
    if not _expand_param_matches(softmax_expand_shape, view_shape):
        raise ValueError(
            f"softmax expand shape {softmax_expand_shape} does not match {view_shape}"
        )
    if output_shape != input_shape:
        raise ValueError(f"final view shape {output_shape} does not match input shape {input_shape}")
    if k_len <= 0:
        raise ValueError("last dimension must be non-empty")
    if k_len > 1024:
        raise ValueError(f"oracle supports K <= 1024 for this repro, got {k_len}")

    return output_shape, batch * heads * q_len, k_len


def _aten_fallback(
    bmm_24: torch.Tensor,
    shape_param_0: Any,
    shape_param_1: Any,
    shape_param_2: Any,
    shape_param_3: Any,
) -> torch.Tensor:
    view_default = torch.ops.aten.view.default(bmm_24, shape_param_0)
    iota_default = torch.ops.prims.iota.default(
        128,
        start=0,
        step=1,
        dtype=torch.int64,
        device=bmm_24.device,
        requires_grad=False,
    )
    add_tensor = torch.ops.aten.add.Tensor(iota_default, 0)
    unsqueeze_default = torch.ops.aten.unsqueeze.default(add_tensor, 0)
    unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1)
    unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3)
    ge_scalar = torch.ops.aten.ge.Scalar(unsqueeze_default_2, 0)
    expand_default = torch.ops.aten.expand.default(ge_scalar, shape_param_1)
    full_default = torch.ops.aten.full.default(
        [],
        0.0,
        dtype=torch.float32,
        layout=torch.strided,
        device=bmm_24.device,
        pin_memory=False,
    )
    full_default_1 = torch.ops.aten.full.default(
        [],
        -float("inf"),
        dtype=torch.float32,
        layout=torch.strided,
        device=bmm_24.device,
        pin_memory=False,
    )
    where_self = torch.ops.aten.where.self(expand_default, full_default, full_default_1)
    add_tensor_1 = torch.ops.aten.add.Tensor(view_default, where_self)
    eq_scalar = torch.ops.aten.eq.Scalar(add_tensor_1, -float("inf"))
    logical_not_default = torch.ops.aten.logical_not.default(eq_scalar)
    any_dim = torch.ops.aten.any.dim(logical_not_default, -1, True)
    logical_not_default_1 = torch.ops.aten.logical_not.default(any_dim)
    full_default_2 = torch.ops.aten.full.default(
        [64, 16, 128, 128],
        0,
        dtype=torch.float32,
        layout=torch.strided,
        device=bmm_24.device,
        pin_memory=False,
    )
    amax_default = torch.ops.aten.amax.default(add_tensor_1, [-1], True)
    sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, amax_default)
    exp_default = torch.ops.aten.exp.default(sub_tensor)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
    div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list)
    where_self_1 = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor)
    expand_default_1 = torch.ops.aten.expand.default(where_self_1, shape_param_2)
    return torch.ops.aten.view.default(expand_default_1, shape_param_3)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=4, num_stages=4),
        ],
        key=["n_rows", "n_cols"],
    )
    @triton.jit
    def oracle_kernel(
        input_ptr,
        output_ptr,
        n_rows: tl.constexpr,
        n_cols: tl.constexpr,
        block_cols: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        col_offsets = tl.arange(0, block_cols)
        mask = (row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols)
        offsets = row_offsets[:, None] * n_cols + col_offsets[None, :]

        scores = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_has_value = tl.sum((mask & (scores != -float("inf"))).to(tl.int32), axis=1) != 0
        row_max = tl.max(scores, axis=1)
        stable_max = tl.where(row_has_value, row_max, 0.0)
        numer = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
        numer = tl.where(mask, numer, 0.0)
        denom = tl.sum(numer, axis=1)
        safe_denom = tl.where(row_has_value, denom, 1.0)
        out_vals = tl.where(row_has_value[:, None], numer / safe_denom[:, None], 0.0)
        tl.store(output_ptr + offsets, out_vals, mask=mask)


def oracle_forward(inputs):
    """Compute exactly Repro()(*make_inputs()) with the full captured scope."""
    bmm_24, shape_param_0, shape_param_1, shape_param_2, shape_param_3 = inputs
    output_shape, n_rows, n_cols = _validate_scope(
        bmm_24,
        shape_param_0,
        shape_param_1,
        shape_param_2,
        shape_param_3,
    )

    if triton is None or not bmm_24.is_cuda or not bmm_24.is_contiguous():
        return _aten_fallback(
            bmm_24,
            shape_param_0,
            shape_param_1,
            shape_param_2,
            shape_param_3,
        )

    output = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=bmm_24.device,
        dtype=torch.float32,
    )
    block_cols = triton.next_power_of_2(n_cols)
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_ROWS"]),)
    oracle_kernel[grid](
        bmm_24,
        output,
        n_rows=n_rows,
        n_cols=n_cols,
        block_cols=block_cols,
    )
    return output


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
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
