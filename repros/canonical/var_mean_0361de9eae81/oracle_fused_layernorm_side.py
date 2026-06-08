"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle hand-codes the complete BEiT affine residual LayerNorm side-output scope in one shape-specialized Triton row kernel, including the metadata-only `[25216, 768] -> [128, 197, 768]` view, `add_76 + arg202_1 * addmm_45`, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)` lowered exactly like generated Inductor as mean then centered squared-difference variance, `libdevice.rsqrt(var + 1e-6)`, affine scale/bias, final contiguous `[25216, 768]` view, and `[128, 197, 1]` `rsqrt * (1 / 768)` side output; Inductor already emits the same fused persistent row-reduction schedule for this fixed hidden-size normalization region, so there is no material algorithmic difference for the oracle to exploit; Inductor cannot materially improve it today through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the remaining cost is dominated by mandatory activation/weight/bias reads, one row reduction, rsqrt, output stores, and launch overhead; the fix is BANDWIDTH_BOUND: record this as a full-scope at-floor LayerNorm side-output case unless broader normalization-template, launch-overhead, or memory-traffic improvements move both paths."""
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

BATCH = 128
TOKENS = 197
HIDDEN = 768
ROWS = BATCH * TOKENS
INPUT_2D_SHAPE = (ROWS, HIDDEN)
INPUT_3D_SHAPE = (BATCH, TOKENS, HIDDEN)
OUTPUT_3D_STRIDE = (TOKENS * HIDDEN, HIDDEN, 1)
OUTPUT_2D_SHAPE = (ROWS, HIDDEN)
OUTPUT_2D_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, TOKENS, 1)
SIDE_STRIDE = (TOKENS, 1, 1)
EPS = 1.0e-6
INV_HIDDEN = 0.0013020833333333333
BLOCK_H = 1024


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _layernorm_side_kernel(
        residual_ptr,
        gamma_ptr,
        addmm_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        side_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask_1d = row_ids < total_rows
        row_mask = row_mask_1d[:, None]
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = rows * hidden + cols

        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        gamma = tl.load(
            gamma_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        addmm = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        bias = tl.load(
            bias_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        )

        scaled = gamma * addmm
        x = residual + scaled
        x_broadcast = tl.broadcast_to(x, [ROW_BLOCK, block_h])
        x_for_reduce = tl.where(mask, x_broadcast, 0)
        mean_sum = tl.sum(x_for_reduce, axis=1)[:, None].to(tl.float32)
        mean_den = (tl.full([1, 1], 768, tl.int32)).to(tl.float32)
        mean = mean_sum / mean_den
        centered_for_var = x_broadcast - mean
        squared = centered_for_var * centered_for_var
        var_sum = tl.sum(tl.where(mask, squared, 0), axis=1)[:, None].to(tl.float32)
        centered = x - mean
        var_den = tl.full([1, 1], 768.0, tl.float32)
        variance = var_sum / var_den
        eps_value = tl.full([1, 1], 1e-06, tl.float32)
        invstd = libdevice.rsqrt(variance + eps_value)
        normalized = centered * invstd
        affine_scaled = normalized * weight
        y = affine_scaled + bias
        side_scale = tl.full([1, 1], 0.0013020833333333333, tl.float32)
        side = invstd * side_scale

        tl.store(out_ptr + offsets, y, mask=mask)
        tl.store(side_ptr + rows, side, mask=row_mask)


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
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_45", inputs[0], INPUT_2D_SHAPE, torch.float32)
    gamma = _require_tensor("arg202_1", inputs[1], (HIDDEN,), torch.float32)
    residual = _require_tensor("add_76", inputs[2], INPUT_3D_SHAPE, torch.float32)
    weight = _require_tensor("arg214_1", inputs[3], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg215_1", inputs[4], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[5]) != INPUT_3D_SHAPE:
        raise ValueError(f"unexpected addmm view shape parameter: {inputs[5]!r}")
    output_shape = _shape_tuple(inputs[6])
    if output_shape != OUTPUT_2D_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {inputs[6]!r}")

    device = addmm.device
    for tensor in (gamma, residual, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same device")
    return addmm, gamma, residual, weight, bias, output_shape


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm, gamma, residual, weight, bias, output_shape = _validate_inputs(inputs)
    viewed = torch.ops.aten.view.default(addmm, _shape_tuple(inputs[5]))
    x = torch.ops.aten.add.Tensor(residual, torch.ops.aten.mul.Tensor(gamma, viewed))
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [2], correction=0, keepdim=True
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(x, mean), invstd)
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, weight), bias)
    return torch.ops.aten.view.default(affine, output_shape), torch.ops.aten.mul.Tensor(invstd, INV_HIDDEN)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the complete Repro.forward affine LayerNorm side-output scope.

    SCOPE INVARIANT: accepts the same seven inputs as Repro.forward() and
    returns the same float32 contiguous `[25216, 768]` view plus float32
    `[128, 197, 1]` side output with stride `(197, 1, 1)`.
    """
    addmm, gamma, residual, weight, bias, output_shape = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output_base = torch.empty_strided(
        INPUT_3D_SHAPE,
        OUTPUT_3D_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _layernorm_side_kernel[grid](
        residual,
        gamma,
        addmm,
        weight,
        bias,
        output_base,
        side,
        total_rows=ROWS,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
    )
    return torch.ops.aten.view.default(output_base, output_shape), side


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    ok = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected, actual)):
        layout_ok = (
            tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
            and actual_tensor.stride() == expected_tensor.stride()
            and actual_tensor.dtype == expected_tensor.dtype
            and actual_tensor.storage_offset() == expected_tensor.storage_offset()
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual_tensor.shape)} stride={actual_tensor.stride()})"
        )
        ok = ok and layout_ok

    view_ok = actual[0]._base is not None
    side_base_ok = actual[1]._base is None
    print(
        f"  output view/base: {'PASS' if view_ok and side_base_ok else 'FAIL'} "
        f"(output0_is_view={view_ok} output1_has_base={actual[1]._base is not None})"
    )
    return ok and view_ok and side_base_ok


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
        layout_ok = _check_layout(instance, inputs)
        print(f"Correctness: {'PASS' if ok and layout_ok else 'FAIL'}")
        if not (ok and layout_ok):
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
