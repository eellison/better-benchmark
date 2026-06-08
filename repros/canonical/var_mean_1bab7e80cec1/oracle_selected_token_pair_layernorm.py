"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DeiT residual-add, fp32 population var_mean(correction=0), `libdevice.rsqrt(var + 1e-6)`, affine LayerNorm, and two token-select outputs by reducing only token indices 0 and 1 for each batch while returning views from a full-stride `[128,198,768]` storage, whereas Inductor lowers the same row-independent LayerNorm producer over all 25,344 token rows before returning two `select(..., 1, {0,1})` views; Inductor cannot do this today because its scheduler does not push fixed token selects backward through view, residual add, var_mean, rsqrt, and affine epilogue to prove the other 196 token rows per batch are dead; the fix is ALGEBRAIC_ELIMINATION: commute constant token selects through row-local normalization graphs and narrow the scheduled row domain before codegen."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


BATCH = 128
TOKENS = 198
HIDDEN = 768
SELECTED_TOKENS = 2
SELECTED_ROWS = BATCH * SELECTED_TOKENS
ROWS = BATCH * TOKENS
ADDMM_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, TOKENS, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
FULL_OUTPUT_SHAPE = RESIDUAL_SHAPE
FULL_OUTPUT_STRIDE = (TOKENS * HIDDEN, HIDDEN, 1)
SELECT_OUTPUT_SHAPE = (BATCH, HIDDEN)
SELECT_OUTPUT_STRIDE = (TOKENS * HIDDEN, 1)
EPS = 1.0e-6
BLOCK_H = 1024
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:
    from torch._inductor.runtime.triton_helpers import libdevice

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 8}, num_warps=8, num_stages=3),
        ],
        key=["hidden", "tokens"],
    )
    @triton.jit
    def _selected_token_pair_layernorm_kernel(
        residual_ptr,
        addmm_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        selected_tokens: tl.constexpr,
        selected_rows_count: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        selected_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, BLOCK)
        row_mask = selected_rows < selected_rows_count
        col_mask = cols < hidden
        mask = row_mask[:, None] & col_mask[None, :]

        batch = selected_rows // selected_tokens
        token = selected_rows - batch * selected_tokens
        source_rows = batch * tokens + token
        offsets = source_rows[:, None] * hidden + cols[None, :]

        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            eviction_policy="evict_first",
            other=0.0,
        )
        addmm = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            eviction_policy="evict_first",
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        bias = tl.load(
            bias_ptr + cols,
            mask=col_mask,
            eviction_policy="evict_last",
            other=0.0,
        )

        x = residual + addmm
        x_for_mean = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_mean, axis=1)[:, None].to(tl.float32) / hidden
        centered_for_var = x - mean
        variance = (
            tl.sum(tl.where(mask, centered_for_var * centered_for_var, 0.0), axis=1)[:, None].to(tl.float32)
            / hidden
        )
        centered = x - mean
        invstd = libdevice.rsqrt(variance + eps)
        normalized = centered * invstd
        scaled = normalized * weight[None, :]
        output = scaled + bias[None, :]

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
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
    expected_stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected_shape}")
    if value.dtype != expected_dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {expected_dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != expected_stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {expected_stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_47 = _require_tensor("addmm_47", inputs[0], ADDMM_SHAPE, torch.float32, (HIDDEN, 1))
    add_80 = _require_tensor("add_80", inputs[1], RESIDUAL_SHAPE, torch.float32, FULL_OUTPUT_STRIDE)
    weight = _require_tensor("arg150_1", inputs[2], AFFINE_SHAPE, torch.float32, (1,))
    bias = _require_tensor("arg151_1", inputs[3], AFFINE_SHAPE, torch.float32, (1,))

    shape_param = _shape_tuple(inputs[4])
    if shape_param != RESIDUAL_SHAPE:
        raise ValueError(f"_shape_param_0 is {shape_param}, expected {RESIDUAL_SHAPE}")

    device = addmm_47.device
    if not (add_80.device == device and weight.device == device and bias.device == device):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm_47, add_80, weight, bias


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same two non-contiguous select views. The full graph is represented by
    direct indexing into the two live token rows per batch: view(addmm_47),
    residual add, fp32 population var_mean over hidden dim, eps-before-rsqrt,
    affine, then select indices 0 and 1 from the sequence dimension.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_selected_token_pair_layernorm.py")

    addmm_47, add_80, weight, bias = _validate_inputs(inputs)
    output_base = torch.empty_strided(
        FULL_OUTPUT_SHAPE,
        FULL_OUTPUT_STRIDE,
        device=addmm_47.device,
        dtype=addmm_47.dtype,
    )
    grid = lambda meta: (triton.cdiv(SELECTED_ROWS, meta["ROW_BLOCK"]),)
    _selected_token_pair_layernorm_kernel[grid](
        add_80,
        addmm_47,
        weight,
        bias,
        output_base,
        hidden=HIDDEN,
        tokens=TOKENS,
        selected_tokens=SELECTED_TOKENS,
        selected_rows_count=SELECTED_ROWS,
        eps=EPS,
        BLOCK=BLOCK_H,
    )
    return (output_base.select(1, 0), output_base.select(1, 1))


def _check_layout_and_aliases(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not (
        isinstance(expected, tuple)
        and isinstance(actual, tuple)
        and len(expected) == 2
        and len(actual) == 2
    ):
        print("  layout/alias: FAIL (output structure mismatch)")
        return False

    ok = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected, actual)):
        layout_ok = (
            tuple(actual_tensor.shape) == SELECT_OUTPUT_SHAPE
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
    print(f"  output alias: {'PASS' if alias_ok else 'FAIL'}")
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
