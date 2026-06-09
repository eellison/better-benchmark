"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BEiT residual-scale add plus LayerNorm scope in one Triton row-reduction kernel using Inductor's centered-variance correction=0 math and libdevice rsqrt, whereas Inductor already emits a single fused persistent reduction kernel with the same required five input reads and one output write; Inductor cannot materially close a local scheduler gap here because the graph is dominated by mandatory residual, scale, addmm, affine-vector, and output memory traffic; the fix is BANDWIDTH_BOUND: record this case as at floor unless broader normalization memory-traffic or launch-overhead improvements move the family."""
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


# --- Oracle kernel(s) ---
BATCH = 128
TOKENS = 197
HIDDEN = 768
ROWS = BATCH * TOKENS
INPUT_SHAPE = (ROWS, HIDDEN)
BASE_SHAPE = (BATCH, TOKENS, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
BASE_STRIDE = (TOKENS * HIDDEN, HIDDEN, 1)
OUTPUT_STRIDE = (HIDDEN, 1)
VECTOR_SHAPE = (HIDDEN,)
VECTOR_STRIDE = (1,)
EPS = 1.0e-6
DTYPE = torch.float32
BLOCK_H = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"XBLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"XBLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["total_rows", "hidden"],
    )
    @triton.jit
    def _scaled_residual_layernorm_kernel(
        addmm_ptr,
        gamma_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        cols = tl.arange(0, BLOCK_H)[None, :]
        row_mask = rows < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = rows * hidden + cols

        addmm = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            eviction_policy="evict_first",
            other=0.0,
        )
        gamma = tl.load(
            gamma_ptr + cols,
            mask=col_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        residual = tl.load(
            residual_ptr + offsets,
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

        scaled = gamma * addmm
        x = residual + scaled

        x_broadcast = tl.broadcast_to(x, [XBLOCK, BLOCK_H])
        x_for_mean = tl.where(mask, x_broadcast, 0.0)
        sum_x = tl.sum(x_for_mean, axis=1)[:, None].to(tl.float32)
        mean_denom = (tl.full([1, 1], 768, tl.int32)).to(tl.float32)
        mean = sum_x / mean_denom

        centered_for_variance = x_broadcast - mean
        square = centered_for_variance * centered_for_variance
        square_broadcast = tl.broadcast_to(square, [XBLOCK, BLOCK_H])
        square_for_sum = tl.where(mask, square_broadcast, 0.0)
        sum_square = tl.sum(square_for_sum, axis=1)[:, None].to(tl.float32)

        centered_for_output = x - mean
        variance_denom = tl.full([1, 1], 768.0, tl.float32)
        variance = sum_square / variance_denom
        eps_value = tl.full([1, 1], 1.0e-6, tl.float32)
        invstd = libdevice.rsqrt(variance + eps_value)

        normalized = centered_for_output * invstd
        weighted = normalized * weight
        out = weighted + bias
        tl.store(out_ptr + offsets, out, mask=mask)


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
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != DTYPE:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {DTYPE}")
    return value


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_1", inputs[0], INPUT_SHAPE, OUTPUT_STRIDE)
    gamma = _require_tensor("arg4_1", inputs[1], VECTOR_SHAPE, VECTOR_STRIDE)
    residual = _require_tensor("cat", inputs[2], BASE_SHAPE, BASE_STRIDE)
    weight = _require_tensor("arg16_1", inputs[3], VECTOR_SHAPE, VECTOR_STRIDE)
    bias = _require_tensor("arg17_1", inputs[4], VECTOR_SHAPE, VECTOR_STRIDE)

    base_shape = _shape_tuple(inputs[5])
    output_shape = _shape_tuple(inputs[6])
    if base_shape != BASE_SHAPE:
        raise ValueError(f"unexpected input view shape parameter: {inputs[5]!r}")
    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"unexpected final view shape parameter: {inputs[6]!r}")

    device = addmm.device
    if any(value.device != device for value in (gamma, residual, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")

    return addmm, gamma, residual, weight, bias, output_shape


def _torch_full_scope(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    addmm, gamma, residual, weight, bias, output_shape = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(addmm, _shape_tuple(inputs[5]))
    mul_tensor = torch.ops.aten.mul.Tensor(gamma, view_default)
    add_tensor = torch.ops.aten.add.Tensor(residual, mul_tensor)
    var, mean = torch.ops.aten.var_mean.correction(
        add_tensor,
        [2],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var, EPS))
    sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, mean)
    normalized = torch.ops.aten.mul.Tensor(sub_tensor, invstd)
    weighted = torch.ops.aten.mul.Tensor(normalized, weight)
    affine = torch.ops.aten.add.Tensor(weighted, bias)
    return torch.ops.aten.view.default(affine, output_shape)


@oracle_impl(hardware="H100", shapes="(T([25216, 768], f32), T([768], f32), T([128, 197, 768], f32), T([768], f32), T([768], f32), S([128, 197, 768]), S([25216, 768]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward scaled-residual LayerNorm computation.

    SCOPE INVARIANT: accepts the same 7 inputs as Repro.forward() and returns
    the same single float32 `[25216, 768]` contiguous view output.
    """
    addmm, gamma, residual, weight, bias, output_shape = _validate_inputs(inputs)
    if triton is None or addmm.device.type != "cuda":
        return _torch_full_scope(inputs)

    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=addmm.device,
        dtype=DTYPE,
    )
    _scaled_residual_layernorm_kernel[(lambda meta: (triton.cdiv(ROWS, meta["XBLOCK"]),))](
        addmm,
        gamma,
        residual,
        weight,
        bias,
        base,
        total_rows=ROWS,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_H=BLOCK_H,
    )
    return base.view(output_shape)


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
