"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Inception inference fragment, including channels-last low-memory max-pool values, both BN-affine ReLU branches, the fixed channel concatenation, and the padded 3x3 stride-1 avg_pool2d materialized in channels-last output layout, whereas Inductor currently schedules the max-pool, pointwise branch producers, cat, and avg-pool consumer as separate materialized regions; Inductor cannot do this today because scheduler fusion does not keep fixed channel-cat operands virtual through structured pooling stencils while preserving channels-last indexing and eliminating the dead max-pool offsets; the fix is SCHEDULER_FUSION: teach the scheduler to sink broadcast pointwise producers and static cat operands into pooling stencil codegen and emit the final layout directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


BATCH = 128
CAT_CHANNELS = 768
CAT_HEIGHT = 17
CAT_WIDTH = 17
BRANCH0_CHANNELS = 320
BRANCH1_CHANNELS = 192
BRANCH_CHANNELS = BRANCH0_CHANNELS + BRANCH1_CHANNELS
OUT_CHANNELS = 1280
HEIGHT = 8
WIDTH = 8
HW = HEIGHT * WIDTH
EPS = 0.001
BLOCK_C = 32
BLOCK_HW = 64

CAT_INPUT_SHAPE = (BATCH, CAT_CHANNELS, CAT_HEIGHT, CAT_WIDTH)
CAT_INPUT_STRIDE = (
    CAT_CHANNELS * CAT_HEIGHT * CAT_WIDTH,
    1,
    CAT_WIDTH * CAT_CHANNELS,
    CAT_CHANNELS,
)
BRANCH0_SHAPE = (BATCH, BRANCH0_CHANNELS, HEIGHT, WIDTH)
BRANCH0_STRIDE = (BRANCH0_CHANNELS * HW, 1, WIDTH * BRANCH0_CHANNELS, BRANCH0_CHANNELS)
BRANCH1_SHAPE = (BATCH, BRANCH1_CHANNELS, HEIGHT, WIDTH)
BRANCH1_STRIDE = (BRANCH1_CHANNELS * HW, 1, WIDTH * BRANCH1_CHANNELS, BRANCH1_CHANNELS)
POOL_VALUE_SHAPE = (BATCH, CAT_CHANNELS, HEIGHT, WIDTH)
POOL_VALUE_STRIDE = (CAT_CHANNELS * HW, 1, WIDTH * CAT_CHANNELS, CAT_CHANNELS)
OUT_SHAPE = (BATCH, OUT_CHANNELS, HEIGHT, WIDTH)
OUT_STRIDE = (OUT_CHANNELS * HW, 1, WIDTH * OUT_CHANNELS, OUT_CHANNELS)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where((x > 0.0) | (x != x), x, 0.0)


    @triton.jit
    def _maxpool_values_kernel(
        cat_ptr,
        values_ptr,
        BLOCK_C_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        c_mask = c_offsets < 768
        hw_mask = hw_offsets < 64
        mask = c_mask[:, None] & hw_mask[None, :]

        oh = hw_offsets // 8
        ow = hw_offsets - oh * 8
        input_base = n * (768 * 17 * 17) + c_offsets[:, None]

        best = tl.full((BLOCK_C_, BLOCK_HW_), -float("inf"), tl.float32)
        for kh in tl.static_range(3):
            ih = oh * 2 + kh
            for kw in tl.static_range(3):
                iw = ow * 2 + kw
                x = tl.load(
                    cat_ptr + input_base + ih[None, :] * (17 * 768) + iw[None, :] * 768,
                    mask=mask,
                    other=-float("inf"),
                    eviction_policy="evict_last",
                ).to(tl.float32)
                best = tl.where(mask & ((x > best) | (x != x)), x, best)

        out_offsets = (
            n * (768 * 64)
            + c_offsets[:, None]
            + oh[None, :] * (8 * 768)
            + ow[None, :] * 768
        )
        tl.store(values_ptr + out_offsets, best, mask=mask)


    @triton.jit
    def _branch_bn_relu_avgpool_kernel(
        mean_ptr,
        conv_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        C: tl.constexpr,
        OUT_C_OFFSET: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        c_mask = c_offsets < C
        hw_mask = hw_offsets < 64
        oh = hw_offsets // 8
        ow = hw_offsets - oh * 8

        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        inv = 1.0 / tl.sqrt(var + 0.001)

        acc = tl.zeros((BLOCK_C_, BLOCK_HW_), dtype=tl.float32)
        input_base = n * C * 64 + c_offsets[:, None]
        for kh in tl.static_range(3):
            ih = oh + kh - 1
            valid_h = (ih >= 0) & (ih < 8)
            for kw in tl.static_range(3):
                iw = ow + kw - 1
                valid = (
                    c_mask[:, None]
                    & hw_mask[None, :]
                    & valid_h[None, :]
                    & (iw[None, :] >= 0)
                    & (iw[None, :] < 8)
                )
                raw = tl.load(
                    conv_ptr + input_base + ih[None, :] * (8 * C) + iw[None, :] * C,
                    mask=valid,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                y = (raw - mean[:, None]) * inv[:, None]
                y = y * weight[:, None] + bias[:, None]
                acc += tl.where(valid, _relu_preserve_nan(y), 0.0)

        out_c = OUT_C_OFFSET + c_offsets[:, None]
        out_offsets = (
            n * (1280 * 64)
            + out_c
            + oh[None, :] * (8 * 1280)
            + ow[None, :] * 1280
        )
        tl.store(
            out_ptr + out_offsets,
            acc * (1.0 / 9.0),
            mask=c_mask[:, None] & hw_mask[None, :],
        )


    @triton.jit
    def _pooled_avgpool_kernel(
        values_ptr,
        out_ptr,
        BLOCK_C_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        c_mask = c_offsets < 768
        hw_mask = hw_offsets < 64
        oh = hw_offsets // 8
        ow = hw_offsets - oh * 8

        acc = tl.zeros((BLOCK_C_, BLOCK_HW_), dtype=tl.float32)
        input_base = n * (768 * 64) + c_offsets[:, None]
        for kh in tl.static_range(3):
            ih = oh + kh - 1
            valid_h = (ih >= 0) & (ih < 8)
            for kw in tl.static_range(3):
                iw = ow + kw - 1
                valid = (
                    c_mask[:, None]
                    & hw_mask[None, :]
                    & valid_h[None, :]
                    & (iw[None, :] >= 0)
                    & (iw[None, :] < 8)
                )
                value = tl.load(
                    values_ptr + input_base + ih[None, :] * (8 * 768) + iw[None, :] * 768,
                    mask=valid,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                acc += tl.where(valid, value, 0.0)

        out_c = 512 + c_offsets[:, None]
        out_offsets = (
            n * (1280 * 64)
            + out_c
            + oh[None, :] * (8 * 1280)
            + ow[None, :] * 1280
        )
        tl.store(
            out_ptr + out_offsets,
            acc * (1.0 / 9.0),
            mask=c_mask[:, None] & hw_mask[None, :],
        )


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects 11 inputs, got {len(inputs)}")

    cat_7 = _require_f32_tensor("cat_7", inputs[0], CAT_INPUT_SHAPE, CAT_INPUT_STRIDE)
    arg357_1 = _require_f32_tensor("arg357_1", inputs[1], (BRANCH0_CHANNELS,), (1,))
    convolution_71 = _require_f32_tensor(
        "convolution_71",
        inputs[2],
        BRANCH0_SHAPE,
        BRANCH0_STRIDE,
    )
    arg358_1 = _require_f32_tensor("arg358_1", inputs[3], (BRANCH0_CHANNELS,), (1,))
    arg359_1 = _require_f32_tensor("arg359_1", inputs[4], (BRANCH0_CHANNELS,), (1,))
    arg360_1 = _require_f32_tensor("arg360_1", inputs[5], (BRANCH0_CHANNELS,), (1,))
    arg377_1 = _require_f32_tensor("arg377_1", inputs[6], (BRANCH1_CHANNELS,), (1,))
    convolution_75 = _require_f32_tensor(
        "convolution_75",
        inputs[7],
        BRANCH1_SHAPE,
        BRANCH1_STRIDE,
    )
    arg378_1 = _require_f32_tensor("arg378_1", inputs[8], (BRANCH1_CHANNELS,), (1,))
    arg379_1 = _require_f32_tensor("arg379_1", inputs[9], (BRANCH1_CHANNELS,), (1,))
    arg380_1 = _require_f32_tensor("arg380_1", inputs[10], (BRANCH1_CHANNELS,), (1,))

    checked = (
        cat_7,
        arg357_1,
        convolution_71,
        arg358_1,
        arg359_1,
        arg360_1,
        arg377_1,
        convolution_75,
        arg378_1,
        arg379_1,
        arg380_1,
    )
    device = cat_7.device
    if any(t.device != device for t in checked):
        raise ValueError("all tensor inputs must be on the same device")
    return checked


def _branch_torch(
    conv: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    y = conv - mean[None, :, None, None]
    y = y * torch.reciprocal(torch.sqrt(var[None, :, None, None] + EPS))
    y = y * weight[None, :, None, None] + bias[None, :, None, None]
    return torch.relu(y)


def _torch_oracle(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    (
        cat_7,
        arg357_1,
        convolution_71,
        arg358_1,
        arg359_1,
        arg360_1,
        arg377_1,
        convolution_75,
        arg378_1,
        arg379_1,
        arg380_1,
    ) = inputs
    values = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        cat_7, [3, 3], [2, 2], [0, 0], [1, 1], False
    )[0]
    branch0 = _branch_torch(convolution_71, arg357_1, arg358_1, arg359_1, arg360_1)
    branch1 = _branch_torch(convolution_75, arg377_1, arg378_1, arg379_1, arg380_1)
    cat = torch.cat([branch0, branch1, values], 1)
    return torch.ops.aten.avg_pool2d.default(cat, [3, 3], [1, 1], [1, 1])


@oracle_impl(hardware="H100", shapes="(T([128, 768, 17, 17], f32, stride=(221952, 1, 13056, 768)), T([320], f32), T([128, 320, 8, 8], f32, stride=(20480, 1, 2560, 320)), T([320], f32), T([320], f32), T([320], f32), T([192], f32), T([128, 192, 8, 8], f32, stride=(12288, 1, 1536, 192)), T([192], f32), T([192], f32), T([192], f32))")
def oracle_forward(inputs):
    """Run the full channels-last maxpool, BN/ReLU, virtual cat, and avgpool scope."""
    checked = _validate_inputs(inputs)
    if not checked[0].is_cuda:
        return _torch_oracle(checked)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    cat_7 = checked[0]
    maxpool_values = torch.empty_strided(
        POOL_VALUE_SHAPE,
        POOL_VALUE_STRIDE,
        device=cat_7.device,
        dtype=torch.float32,
    )
    avg = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=cat_7.device,
        dtype=torch.float32,
    )

    maxpool_grid = (BATCH, triton.cdiv(CAT_CHANNELS, BLOCK_C))
    _maxpool_values_kernel[maxpool_grid](
        cat_7,
        maxpool_values,
        BLOCK_C_=BLOCK_C,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )

    branch0_grid = (BATCH, triton.cdiv(BRANCH0_CHANNELS, BLOCK_C))
    _branch_bn_relu_avgpool_kernel[branch0_grid](
        checked[1],
        checked[2],
        checked[3],
        checked[4],
        checked[5],
        avg,
        C=BRANCH0_CHANNELS,
        OUT_C_OFFSET=0,
        BLOCK_C_=BLOCK_C,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )

    branch1_grid = (BATCH, triton.cdiv(BRANCH1_CHANNELS, BLOCK_C))
    _branch_bn_relu_avgpool_kernel[branch1_grid](
        checked[6],
        checked[7],
        checked[8],
        checked[9],
        checked[10],
        avg,
        C=BRANCH1_CHANNELS,
        OUT_C_OFFSET=BRANCH0_CHANNELS,
        BLOCK_C_=BLOCK_C,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )

    pooled_grid = (BATCH, triton.cdiv(CAT_CHANNELS, BLOCK_C))
    _pooled_avgpool_kernel[pooled_grid](
        maxpool_values,
        avg,
        BLOCK_C_=BLOCK_C,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )
    return avg


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def check_oracle_equal_nan(
    oracle_forward_fn,
    instance,
    inputs,
    *,
    atol: float,
    rtol: float,
    skip_stochastic: bool,
) -> bool:
    """Correctness check matching the harness while accepting deterministic NaNs."""
    if skip_stochastic and has_stochastic_ops(REPRO_PATH):
        print("  stochastic output skipping requested, but this repro has no stochastic outputs")

    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward_fn(inputs)

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)
    if len(oracle_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for idx, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        layout_ok = (
            tuple(actual.shape) == tuple(expected.shape)
            and tuple(actual.stride()) == tuple(expected.stride())
            and actual.dtype == expected.dtype
        )
        print(
            f"  output {idx} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual.shape)} stride={tuple(actual.stride())} "
            f"dtype={actual.dtype})"
        )
        all_pass = all_pass and layout_ok
        if not layout_ok:
            continue

        if not expected.is_floating_point():
            ok = torch.equal(expected, actual)
            print(f"  output {idx}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected.dtype})")
            all_pass = all_pass and ok
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        finite = torch.isfinite(expected_f32) & torch.isfinite(actual_f32)
        if finite.any():
            max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
        else:
            max_diff = 0.0
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_mismatch = torch.logical_xor(expected_nan, actual_nan).sum().item()
        ok = torch.allclose(
            expected_f32,
            actual_f32,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
        print(
            f"  output {idx}: {'PASS' if ok else 'FAIL'} "
            f"(finite_max_diff={max_diff:.2e} nan_count={expected_nan.sum().item()} "
            f"nan_mismatch={nan_mismatch})"
        )
        all_pass = all_pass and ok

    return all_pass


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
        ok = check_oracle_equal_nan(
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
