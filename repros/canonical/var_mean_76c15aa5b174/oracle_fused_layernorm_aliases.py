"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BART-style residual LayerNorm alias scope in one shape-specialized Triton row kernel, including the metadata-only addmm view, residual add, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)` lowered as generated Inductor does with mean followed by centered squared-difference variance, `libdevice.rsqrt(var + 1e-5)`, affine multiply/add, and three returned `[rows, hidden]` alias views over one contiguous base allocation, whereas Inductor already emits one fused persistent inner-reduction kernel for the same full scope; Inductor cannot materially improve this repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a narrower pattern because the remaining work is the required activation/residual/affine reads, one row reduction, reciprocal square root, and output store; the fix is BANDWIDTH_BOUND: record this as a full-scope normalization floor unless broader norm-template, launch-overhead, or memory-traffic improvements move both implementations."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


EPS = 1.0e-5


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= int(size)
    return tuple(reversed(strides))


def _require_f32_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int],
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    addmm, residual, weight, bias, shape0, shape1, shape2, shape3 = inputs
    addmm_t = _require_f32_tensor("addmm_65", addmm)
    residual_t = _require_f32_tensor("add_79", residual)
    weight_t = _require_f32_tensor("arg179_1", weight)
    bias_t = _require_f32_tensor("arg180_1", bias)

    if addmm_t.ndim != 2:
        raise ValueError(f"addmm_65 must be rank-2, got shape={tuple(addmm_t.shape)}")
    if residual_t.ndim != 3:
        raise ValueError(f"add_79 must be rank-3, got shape={tuple(residual_t.shape)}")
    if not addmm_t.is_contiguous():
        raise ValueError(f"addmm_65 must be contiguous, got stride={tuple(addmm_t.stride())}")
    if not residual_t.is_contiguous():
        raise ValueError(f"add_79 must be contiguous, got stride={tuple(residual_t.stride())}")
    if not weight_t.is_contiguous() or not bias_t.is_contiguous():
        raise ValueError("arg179_1 and arg180_1 must be contiguous")

    rows = int(addmm_t.shape[0])
    hidden = int(addmm_t.shape[1])
    base_shape = _shape_tuple(shape0)
    output_shapes = (_shape_tuple(shape1), _shape_tuple(shape2), _shape_tuple(shape3))

    if tuple(residual_t.shape) != base_shape:
        raise ValueError(
            f"_shape_param_0 {base_shape!r} does not match add_79 shape "
            f"{tuple(residual_t.shape)}"
        )
    if len(base_shape) != 3 or base_shape[0] * base_shape[1] != rows or base_shape[2] != hidden:
        raise ValueError(
            f"_shape_param_0 {base_shape!r} is incompatible with addmm_65 "
            f"shape {tuple(addmm_t.shape)}"
        )
    if tuple(weight_t.shape) != (hidden,) or tuple(bias_t.shape) != (hidden,):
        raise ValueError(
            f"arg179_1/arg180_1 must both have shape {(hidden,)}, got "
            f"{tuple(weight_t.shape)} and {tuple(bias_t.shape)}"
        )
    for index, shape in enumerate(output_shapes, start=1):
        if shape != (rows, hidden):
            raise ValueError(f"_shape_param_{index} is {shape!r}, expected {(rows, hidden)}")
    if any(tensor.device != addmm_t.device for tensor in (residual_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same device")

    return addmm_t, residual_t, weight_t, bias_t, base_shape, output_shapes


def _block_hidden(hidden: int) -> int:
    return 1 << (hidden - 1).bit_length()


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _fused_layernorm_aliases_kernel(
        residual_ptr,
        addmm_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * row_block + tl.arange(0, row_block)
        rows = row_ids[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = row_ids[:, None] < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = rows * hidden + cols

        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
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

        x = residual + addmm
        x_broadcast = tl.broadcast_to(x, [row_block, block_h])
        sum_x = tl.sum(tl.where(mask, x_broadcast, 0.0), axis=1)[:, None].to(tl.float32)
        mean_denom = (tl.full([1, 1], hidden, tl.int32)).to(tl.float32)
        mean = sum_x / mean_denom
        centered_for_var = x_broadcast - mean
        sum_sq = tl.sum(
            tl.where(mask, centered_for_var * centered_for_var, 0.0),
            axis=1,
        )[:, None].to(tl.float32)
        var_denom = tl.full([1, 1], hidden + 0.0, tl.float32)
        variance = sum_sq / var_denom
        eps_value = tl.full([1, 1], eps, tl.float32)
        invstd = libdevice.rsqrt(variance + eps_value)

        centered = x - mean
        normalized = centered * invstd
        scaled = normalized * weight
        output = scaled + bias
        tl.store(output_ptr + offsets, output, mask=mask)


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    addmm, residual, weight, bias, base_shape, output_shapes = _validate_inputs(inputs)
    viewed = torch.ops.aten.view.default(addmm, base_shape)
    added = torch.ops.aten.add.Tensor(residual, viewed)
    variance, mean = torch.ops.aten.var_mean.correction(
        added, [2], correction=0, keepdim=True
    )
    centered = torch.ops.aten.sub.Tensor(added, mean)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(centered, invstd)
    scaled = torch.ops.aten.mul.Tensor(normalized, weight)
    output = torch.ops.aten.add.Tensor(scaled, bias)
    return tuple(torch.ops.aten.view.default(output, shape) for shape in output_shapes)


def oracle_forward(inputs):
    """Run the complete Repro.forward computation.

    SCOPE INVARIANT: accepts the same eight inputs as Repro.forward() and
    returns the same three fp32 [rows, hidden] contiguous views, all aliasing
    one normalized base allocation.
    """
    addmm, residual, weight, bias, base_shape, output_shapes = _validate_inputs(inputs)
    if triton is None or libdevice is None or addmm.device.type != "cuda":
        return _torch_full_scope(inputs)

    rows = int(addmm.shape[0])
    hidden = int(addmm.shape[1])
    block_h = _block_hidden(hidden)
    if block_h > 4096:
        raise ValueError(f"hidden size {hidden} is too large for this oracle")
    row_block = 2 if hidden <= 1024 else 1
    num_warps = 4 if hidden <= 1024 else 8

    output_base = torch.empty_strided(
        base_shape,
        _contiguous_strides(base_shape),
        device=addmm.device,
        dtype=torch.float32,
    )
    _fused_layernorm_aliases_kernel[(triton.cdiv(rows, row_block),)](
        residual,
        addmm,
        weight,
        bias,
        output_base,
        total_rows=rows,
        hidden=hidden,
        eps=EPS,
        block_h=block_h,
        row_block=row_block,
        num_warps=num_warps,
        num_stages=1,
    )
    return tuple(torch.ops.aten.view.default(output_base, shape) for shape in output_shapes)


def _normalize_outputs(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _layout_signature(value: torch.Tensor) -> tuple[torch.dtype, tuple[int, ...], tuple[int, ...], int]:
    return value.dtype, tuple(value.shape), tuple(value.stride()), int(value.storage_offset())


def _check_layout_alias_and_diff(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        expected = _normalize_outputs(instance(*inputs))
        actual = _normalize_outputs(oracle_forward(inputs))
        if any(isinstance(item, torch.Tensor) and item.device.type == "cuda" for item in actual):
            torch.cuda.synchronize()

    ok = True
    max_diffs: list[float] = []
    for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
        if not isinstance(expected_item, torch.Tensor) or not isinstance(actual_item, torch.Tensor):
            continue
        layout_ok = _layout_signature(expected_item) == _layout_signature(actual_item)
        diff = (expected_item.float() - actual_item.float()).abs().max().item()
        max_diffs.append(diff)
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(expected={_layout_signature(expected_item)} oracle={_layout_signature(actual_item)})"
        )
        ok = layout_ok and ok

    expected_tensors = [item for item in expected if isinstance(item, torch.Tensor)]
    actual_tensors = [item for item in actual if isinstance(item, torch.Tensor)]
    expected_alias = (
        len(expected_tensors) == 3
        and len({tensor.untyped_storage().data_ptr() for tensor in expected_tensors}) == 1
        and len({int(tensor.storage_offset()) for tensor in expected_tensors}) == 1
    )
    actual_alias = (
        len(actual_tensors) == 3
        and len({tensor.untyped_storage().data_ptr() for tensor in actual_tensors}) == 1
        and len({int(tensor.storage_offset()) for tensor in actual_tensors}) == 1
    )
    alias_ok = expected_alias == actual_alias
    diff_text = ", ".join(f"{value:.2e}" for value in max_diffs)
    print(
        f"  output alias: {'PASS' if alias_ok else 'FAIL'} "
        f"(expected={expected_alias} oracle={actual_alias}; max_diffs=[{diff_text}])"
    )
    return ok and alias_ok


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
        layout_ok = _check_layout_alias_and_diff(instance, inputs)
        ok = ok and layout_ok
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
