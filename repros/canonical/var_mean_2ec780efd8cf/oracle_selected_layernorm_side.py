"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete residual-add hidden-size-768 LayerNorm scope, including all-row fp32 correction=0 var_mean statistics, Inductor's eps-before-libdevice.rsqrt lowering, the live `rsqrt / 768` side output, and the two non-contiguous selected token views sharing one backing buffer, whereas Inductor currently materializes the full `[128,198,768]` affine tensor before returning only token 0 and token 1; Inductor cannot do this today because its normalization epilogue liveness does not push final select users into the row-reduction store plan while preserving alias-view metadata and the all-row side output; the fix is ALGEBRAIC_ELIMINATION: split the normalization epilogue into all-row side stores plus selected-row affine stores, with the returned selects represented as alias views over the same backing storage."""
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
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None
    libdevice = None

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
    has_stochastic_ops,
)


BATCH = 128
TOKENS = 198
HIDDEN = 768
N_ROWS = BATCH * TOKENS

ADDMM_SHAPE = (N_ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, TOKENS, HIDDEN)
WEIGHT_SHAPE = (HIDDEN,)
BIAS_SHAPE = (HIDDEN,)
OUTPUT_BASE_SHAPE = RESIDUAL_SHAPE
OUTPUT_BASE_STRIDE = (TOKENS * HIDDEN, HIDDEN, 1)
SIDE_SHAPE = (BATCH, TOKENS, 1)
SIDE_STRIDE = (TOKENS, 1, 1)

EPS = 1.0e-6
INV_HIDDEN = 1.0 / HIDDEN
BLOCK_H = 1024


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
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
    def _selected_layernorm_side_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        side_ptr,
        total_rows: tl.constexpr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        inv_hidden: tl.constexpr,
        block_h: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, block_h)
        row_mask = row_ids < total_rows
        col_mask = cols < hidden
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = row_ids[:, None] * hidden + cols[None, :]

        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        addmm = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = residual + addmm

        x_for_reduce = tl.where(mask, x, 0.0)
        sum_x = tl.sum(x_for_reduce, axis=1)[:, None].to(tl.float32)
        mean_denominator = tl.full([1, 1], hidden, tl.int32).to(tl.float32)
        mean = sum_x / mean_denominator

        centered_for_var = x - mean
        square = centered_for_var * centered_for_var
        variance_sum = tl.sum(tl.where(mask, square, 0.0), axis=1)[:, None].to(tl.float32)
        variance_denominator = tl.full([1, 1], hidden, tl.float32)
        variance = variance_sum / variance_denominator
        eps_value = tl.full([1, 1], eps, tl.float32)
        invstd = libdevice.rsqrt(variance + eps_value)
        inv_hidden_value = tl.full([1, 1], inv_hidden, tl.float32)

        tl.store(
            side_ptr + row_ids[:, None],
            invstd * inv_hidden_value,
            mask=row_mask[:, None],
        )

        token_ids = row_ids % tokens
        selected = token_ids < 2
        param_offsets = tl.broadcast_to(cols[None, :], [ROW_BLOCK, block_h])
        param_mask = selected[:, None] & col_mask[None, :]
        weight = tl.load(
            weight_ptr + param_offsets,
            mask=param_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + param_offsets,
            mask=param_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        normalized = (x - mean) * invstd
        out = normalized * weight + bias
        tl.store(out_ptr + offsets, out, mask=mask & selected[:, None])


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
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm, residual, weight, bias, shape0 = inputs
    addmm = _require_tensor("addmm_47", addmm, ADDMM_SHAPE, (HIDDEN, 1), torch.float32)
    residual = _require_tensor(
        "add_80",
        residual,
        RESIDUAL_SHAPE,
        OUTPUT_BASE_STRIDE,
        torch.float32,
    )
    weight = _require_tensor("arg150_1", weight, WEIGHT_SHAPE, (1,), torch.float32)
    bias = _require_tensor("arg151_1", bias, BIAS_SHAPE, (1,), torch.float32)

    shape0_tuple = _shape_tuple(shape0)
    if shape0_tuple != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected view shape parameter: {shape0!r}")

    device = addmm.device
    if any(tensor.device != device for tensor in (residual, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")

    return addmm, residual, weight, bias, shape0_tuple


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    addmm, residual, weight, bias, shape0 = _validate_inputs(inputs)
    viewed = torch.ops.aten.view.default(addmm, shape0)
    x = torch.ops.aten.add.Tensor(residual, viewed)
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [2], correction=0, keepdim=True
    )
    invstd = torch.ops.aten.rsqrt.default(variance + EPS)
    normalized = torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(x, mean), invstd)
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, weight), bias)
    return (
        torch.ops.aten.select.int(affine, 1, 0),
        torch.ops.aten.select.int(affine, 1, 1),
        torch.ops.aten.div.Tensor(invstd, HIDDEN),
    )


@oracle_impl(hardware="H100", shapes="(T([25344, 768], f32), T([128, 198, 768], f32), T([768], f32), T([768], f32), S([128, 198, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward residual LayerNorm select-plus-side scope."""
    addmm, residual, weight, bias, _shape0 = _validate_inputs(inputs)
    if triton is None or libdevice is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output_base = torch.empty_strided(
        OUTPUT_BASE_SHAPE,
        OUTPUT_BASE_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(N_ROWS, meta["ROW_BLOCK"]),)
    _selected_layernorm_side_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output_base,
        side,
        total_rows=N_ROWS,
        tokens=TOKENS,
        hidden=HIDDEN,
        eps=EPS,
        inv_hidden=INV_HIDDEN,
        block_h=BLOCK_H,
    )
    return output_base.select(1, 0), output_base.select(1, 1), side


def _check_layout_and_aliases(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not (
        isinstance(expected, tuple)
        and isinstance(actual, tuple)
        and len(expected) == 3
        and len(actual) == 3
    ):
        print("  layout/alias: FAIL (output structure mismatch)")
        return False

    ok = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected, actual)):
        layout_ok = (
            tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
            and actual_tensor.dtype == expected_tensor.dtype
            and actual_tensor.stride() == expected_tensor.stride()
            and actual_tensor.storage_offset() == expected_tensor.storage_offset()
            and actual_tensor.untyped_storage().nbytes() == expected_tensor.untyped_storage().nbytes()
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual_tensor.shape)} stride={actual_tensor.stride()} "
            f"offset={actual_tensor.storage_offset()})"
        )
        ok = ok and layout_ok

    expected_alias = expected[0].untyped_storage().data_ptr() == expected[1].untyped_storage().data_ptr()
    actual_alias = actual[0].untyped_storage().data_ptr() == actual[1].untyped_storage().data_ptr()
    alias_ok = expected_alias and actual_alias
    print(f"  selected-output alias: {'PASS' if alias_ok else 'FAIL'}")
    return ok and alias_ok


# --- CLI entry point ---
def main() -> None:
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
        layout_ok = _check_layout_and_aliases(instance, inputs)
        print(f"Correctness: {'PASS' if ok and layout_ok else 'FAIL'}")
        if not (ok and layout_ok):
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
