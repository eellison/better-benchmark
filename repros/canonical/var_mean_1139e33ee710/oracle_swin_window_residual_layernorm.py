"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin window-reverse residual LayerNorm scope in one shape-specialized Triton row kernel, preserving the generated population Welford var_mean, eps-before-rsqrt placement, libdevice.rsqrt lowering, affine epilogue, and final contiguous `[25088,512]` view, whereas Inductor currently emits a fused generic reduction kernel that reloads the window-reversed residual row for the normalization epilogue after computing Welford statistics; Inductor cannot do this today because its normalization scheduler does not retain a nontrivial reshape/permute/clone producer tile across the row-statistics pass and affine epilogue; the fix is SCHEDULER_FUSION: extend the fixed-hidden LayerNorm template to inline window-reverse producers and carry the row tile into the epilogue when the producer is single-use."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    triton_helpers = None
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


BATCH = 128
HEIGHT = 14
WIDTH = 14
PATCHES = HEIGHT * WIDTH
HIDDEN = 512
ROWS = BATCH * PATCHES
WINDOW = 7
WINDOW_AREA = WINDOW * WINDOW
WINDOWS_H = HEIGHT // WINDOW
WINDOWS_W = WIDTH // WINDOW
WINDOWS_PER_IMAGE = WINDOWS_H * WINDOWS_W
EPS = 1.0e-5
DTYPE = torch.float32

ADDMM_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, HEIGHT, WIDTH, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
EXPECTED_SHAPE_PARAMS = (
    (WINDOWS_PER_IMAGE * BATCH, WINDOW_AREA, HIDDEN),
    (-1, WINDOW, WINDOW, HIDDEN),
    (-1, WINDOWS_H, WINDOWS_W, WINDOW, WINDOW, HIDDEN),
    (-1, HEIGHT, WIDTH, HIDDEN),
    (BATCH, -1, HIDDEN),
    OUTPUT_SHAPE,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(name: str, value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a shape sequence, got {value!r}") from exc


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
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_81", inputs[0], ADDMM_SHAPE, DTYPE)
    residual = _require_tensor("view_545", inputs[1], RESIDUAL_SHAPE, DTYPE)
    weight = _require_tensor("arg309_1", inputs[2], AFFINE_SHAPE, DTYPE)
    bias = _require_tensor("arg310_1", inputs[3], AFFINE_SHAPE, DTYPE)

    device = addmm.device
    for index, tensor in enumerate((residual, weight, bias), start=1):
        if tensor.device != device:
            raise ValueError(f"input {index} device {tensor.device} != {device}")

    shape_params = tuple(
        _shape_tuple(f"_shape_param_{index}", value)
        for index, value in enumerate(inputs[4:])
    )
    if shape_params != EXPECTED_SHAPE_PARAMS:
        raise ValueError(
            f"unexpected shape params {shape_params!r}, expected {EXPECTED_SHAPE_PARAMS!r}"
        )

    return addmm, residual, weight, bias, shape_params[-1]


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _swin_window_residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        patches: tl.constexpr,
        width: tl.constexpr,
        window: tl.constexpr,
        windows_w: tl.constexpr,
        window_area: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, BLOCK_H)[None, :]
        row_mask = row_ids[:, None] < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask

        patch = rows % patches
        batch = rows // patches
        h = patch // width
        w = patch - h * width
        window_h = h // window
        window_w = w // window
        inner_h = h - window_h * window
        inner_w = w - window_w * window

        addmm_offsets = (
            batch * patches * hidden
            + (window_h * windows_w + window_w) * window_area * hidden
            + (inner_h * window + inner_w) * hidden
            + cols
        )
        flat_offsets = rows * hidden + cols

        residual = tl.load(
            residual_ptr + flat_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        projected = tl.load(
            addmm_ptr + addmm_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = residual + projected

        mean_state = tl.zeros([ROW_BLOCK, BLOCK_H], tl.float32)
        m2_state = tl.zeros([ROW_BLOCK, BLOCK_H], tl.float32)
        weight_state = tl.zeros([ROW_BLOCK, BLOCK_H], tl.float32)
        mean_state, m2_state, weight_state = triton_helpers.welford_reduce(
            x, mean_state, m2_state, weight_state, True
        )
        mean, m2, _count = triton_helpers.welford(
            mean_state, m2_state, weight_state, 1
        )
        mean = mean[:, None]
        variance = m2[:, None] / tl.full([1, 1], 512.0, tl.float32)
        invstd = libdevice.rsqrt(variance + tl.full([1, 1], eps, tl.float32))

        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        output = (x - mean) * invstd * weight + bias
        tl.store(output_ptr + flat_offsets, output, mask=mask)


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias, output_shape = _validate_inputs(inputs)
    x = torch.ops.aten.view.default(addmm, _shape_tuple("_shape_param_0", inputs[4]))
    x = torch.ops.aten.view.default(x, _shape_tuple("_shape_param_1", inputs[5]))
    x = torch.ops.aten.view.default(x, _shape_tuple("_shape_param_2", inputs[6]))
    x = torch.ops.aten.permute.default(x, [0, 1, 3, 2, 4, 5])
    x = torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)
    x = torch.ops.aten.view.default(x, _shape_tuple("_shape_param_3", inputs[7]))
    x = torch.ops.aten.add.Tensor(residual, x)
    x = torch.ops.aten.view.default(x, _shape_tuple("_shape_param_4", inputs[8]))
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [2], correction=0, keepdim=True
    )
    centered = torch.ops.aten.sub.Tensor(x, mean)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(centered, invstd)
    scaled = torch.ops.aten.mul.Tensor(normalized, weight)
    biased = torch.ops.aten.add.Tensor(scaled, bias)
    return torch.ops.aten.view.default(biased, output_shape)


@oracle_impl(hardware="H100", shapes="(T([25088, 512], f32), T([128, 14, 14, 512], f32), T([512], f32), T([512], f32), S([512, 49, 512]), S([-1, 7, 7, 512]), S([-1, 2, 2, 7, 7, 512]), S([-1, 14, 14, 512]), S([128, -1, 512]), S([25088, 512]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    addmm, residual, weight, bias, output_shape = _validate_inputs(inputs)
    if (
        triton is None
        or triton_helpers is None
        or libdevice is None
        or not addmm.is_cuda
    ):
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        output_shape,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=DTYPE,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _swin_window_residual_layernorm_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output,
        total_rows=ROWS,
        patches=PATCHES,
        width=WIDTH,
        window=WINDOW,
        windows_w=WINDOWS_W,
        window_area=WINDOW_AREA,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_H=HIDDEN,
    )
    return output


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
