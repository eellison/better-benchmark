"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Longformer inference bias-add residual LayerNorm scope in one shape-specialized Triton row kernel, including the `[8192,768] -> [8,1024,768]` metadata view, `(mm_47 + arg186_1) + add_131`, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)` as Inductor's generated mean plus centered squared-difference variance, `libdevice.rsqrt(var + 1e-5)`, affine scale/bias, and final contiguous `[8192,768]` view, whereas Inductor already emits one fused persistent row-reduction kernel for this exact full scope; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a new pattern because the remaining cost is dominated by mandatory activation/residual/affine reads, one hidden-size-768 reduction, rsqrt, and output stores; the fix is BANDWIDTH_BOUND: record this as an at-floor LayerNorm case unless broader normalization-template, launch-overhead, or memory-traffic work moves both paths."""
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


BATCH = 8
SEQ_LEN = 1024
HIDDEN = 768
ROWS = BATCH * SEQ_LEN
INPUT_SHAPE = (ROWS, HIDDEN)
VIEW_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
VIEW_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
VECTOR_SHAPE = (HIDDEN,)
EPS = 1.0e-5
BLOCK_H = 1024
CLASSIFICATION = "BANDWIDTH_BOUND"


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
            triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _bias_residual_layernorm_kernel(
        mm_ptr,
        pre_bias_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        block_h: tl.constexpr,
        total_rows: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = row_ids[:, None] < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = rows * hidden + cols

        mm = tl.load(
            mm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        pre_bias = tl.load(
            pre_bias_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        residual = tl.load(
            residual_ptr + offsets,
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

        add_bias = mm + pre_bias
        x = add_bias + residual
        x_broadcast = tl.broadcast_to(x, [ROW_BLOCK, block_h])
        x_for_reduce = tl.where(mask, x_broadcast, 0.0)
        mean_sum = tl.sum(x_for_reduce, axis=1)[:, None].to(tl.float32)
        mean_den = tl.full([1, 1], 768, tl.int32).to(tl.float32)
        mean = mean_sum / mean_den

        centered_for_var = x_broadcast - mean
        squared = centered_for_var * centered_for_var
        var_sum = tl.sum(tl.where(mask, squared, 0.0), axis=1)[:, None].to(tl.float32)
        centered = x - mean
        var_den = tl.full([1, 1], 768.0, tl.float32)
        variance = var_sum / var_den
        eps_value = tl.full([1, 1], 1e-05, tl.float32)
        invstd = libdevice.rsqrt(variance + eps_value)

        normalized = centered * invstd
        scaled = normalized * weight
        output = scaled + bias
        tl.store(out_ptr + offsets, output, mask=mask)


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
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    mm_47 = _require_tensor("mm_47", inputs[0], INPUT_SHAPE, torch.float32, OUTPUT_STRIDE)
    pre_bias = _require_tensor("arg186_1", inputs[1], VECTOR_SHAPE, torch.float32, (1,))
    residual = _require_tensor("add_131", inputs[2], VIEW_SHAPE, torch.float32, VIEW_STRIDE)
    weight = _require_tensor("arg187_1", inputs[3], VECTOR_SHAPE, torch.float32, (1,))
    bias = _require_tensor("arg188_1", inputs[4], VECTOR_SHAPE, torch.float32, (1,))

    if _shape_tuple(inputs[5]) != VIEW_SHAPE:
        raise ValueError(f"unexpected input view shape parameter: {inputs[5]!r}")
    output_shape = _shape_tuple(inputs[6])
    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {inputs[6]!r}")

    device = mm_47.device
    if any(value.device != device for value in (pre_bias, residual, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return mm_47, pre_bias, residual, weight, bias, output_shape


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    mm_47, pre_bias, residual, weight, bias, output_shape = _validate_inputs(inputs)
    viewed = torch.ops.aten.view.default(mm_47, VIEW_SHAPE)
    add_tensor = torch.ops.aten.add.Tensor(viewed, pre_bias)
    x = torch.ops.aten.add.Tensor(add_tensor, residual)
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [2], correction=0, keepdim=True
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(x, mean), invstd)
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, weight), bias)
    return torch.ops.aten.view.default(affine, output_shape)


@oracle_impl(hardware="H100", shapes="(T([8192, 768], f32), T([768], f32), T([8, 1024, 768], f32), T([768], f32), T([768], f32), S([8, 1024, 768]), S([8192, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete Repro.forward bias-residual LayerNorm scope.

    The Triton path preserves the generated Inductor math: f32 `(mm + bias) +
    residual`, mean then centered population variance, epsilon before
    `libdevice.rsqrt`, f32 affine epilogue, and a final `[8192, 768]` view over
    the contiguous `[8, 1024, 768]` output allocation.
    """
    mm_47, pre_bias, residual, weight, bias, output_shape = _validate_inputs(inputs)
    if triton is None or libdevice is None or not mm_47.is_cuda:
        return _torch_full_scope(inputs)

    output_base = torch.empty_strided(
        VIEW_SHAPE,
        VIEW_STRIDE,
        device=mm_47.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _bias_residual_layernorm_kernel[grid](
        mm_47,
        pre_bias,
        residual,
        weight,
        bias,
        output_base,
        hidden=HIDDEN,
        block_h=BLOCK_H,
        total_rows=ROWS,
    )
    return torch.ops.aten.view.default(output_base, output_shape)


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    layout_ok = (
        tuple(actual.shape) == tuple(expected.shape)
        and tuple(actual.stride()) == tuple(expected.stride())
        and actual.dtype == expected.dtype
        and actual.storage_offset() == expected.storage_offset()
    )
    view_ok = (actual._base is not None) == (expected._base is not None)
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={actual.stride()})"
    )
    print(
        f"  output 0 view/base: {'PASS' if view_ok else 'FAIL'} "
        f"(oracle_is_view={actual._base is not None} eager_is_view={expected._base is not None})"
    )
    return layout_ok and view_ok


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
        ok = _check_layout(instance, inputs) and ok
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
