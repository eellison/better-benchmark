"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Inception low-memory maxpool-with-offsets, two BN-inference ReLU branches, fixed channel concatenation, and padded 3x3 stride-1 avg_pool2d scope with branch-specialized Triton stencil kernels while returning the exact int8 offsets and final f32 `[128,1280,8,8]` tensor, whereas Inductor currently schedules the maxpool, both BN/ReLU branch producers, and the avgpool consumer as separate generic regions with materialized branch activations; Inductor cannot do this today because scheduler fusion does not keep fixed channel-cat operands virtual through structured pooling consumers while also preserving the sibling maxpool-offset output; the fix is SCHEDULER_FUSION: teach the scheduler to route static cat operands from broadcast pointwise producers and required maxpool staging into branch-specialized avgpool stencil kernels."""
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
    check_oracle,
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
AVG_CHANNELS = 1280
HEIGHT = 8
WIDTH = 8
HW = HEIGHT * WIDTH
EPS = 0.001
BLOCK_C = 4
BLOCK_HW = 64


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
    def _maxpool_values_offsets_kernel(
        cat_ptr,
        values_ptr,
        offsets_ptr,
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
        input_base = (n * 768 + c_offsets[:, None]) * 289

        best = tl.full((BLOCK_C_, BLOCK_HW_), -float("inf"), tl.float32)
        best_offset = tl.zeros((BLOCK_C_, BLOCK_HW_), tl.int32)

        for kh in tl.static_range(3):
            ih = oh * 2 + kh
            for kw in tl.static_range(3):
                iw = ow * 2 + kw
                x = tl.load(
                    cat_ptr + input_base + ih[None, :] * 17 + iw[None, :],
                    mask=mask,
                    other=-float("inf"),
                    eviction_policy="evict_last",
                ).to(tl.float32)
                take = mask & ((x > best) | (x != x))
                best = tl.where(take, x, best)
                best_offset = tl.where(take, kh * 3 + kw, best_offset)

        out_base = (n * 768 + c_offsets[:, None]) * 64
        tl.store(values_ptr + out_base + hw_offsets[None, :], best, mask=mask)
        tl.store(
            offsets_ptr + out_base + hw_offsets[None, :],
            best_offset.to(tl.int8),
            mask=mask,
        )


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
        input_base = (n * C + c_offsets[:, None]) * 64
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
                    conv_ptr + input_base + ih[None, :] * 8 + iw[None, :],
                    mask=valid,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                y = (raw - mean[:, None]) * inv[:, None]
                y = y * weight[:, None] + bias[:, None]
                acc += tl.where(valid, _relu_preserve_nan(y), 0.0)

        out_c = OUT_C_OFFSET + c_offsets[:, None]
        out_base = (n * 1280 + out_c) * 64
        tl.store(
            out_ptr + out_base + hw_offsets[None, :],
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
        input_base = (n * 768 + c_offsets[:, None]) * 64
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
                    values_ptr + input_base + ih[None, :] * 8 + iw[None, :],
                    mask=valid,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                acc += tl.where(valid, value, 0.0)

        out_c = 512 + c_offsets[:, None]
        out_base = (n * 1280 + out_c) * 64
        tl.store(
            out_ptr + out_base + hw_offsets[None, :],
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

    cat_7 = _require_f32_tensor(
        "cat_7",
        inputs[0],
        (BATCH, CAT_CHANNELS, CAT_HEIGHT, CAT_WIDTH),
        (CAT_CHANNELS * CAT_HEIGHT * CAT_WIDTH, CAT_HEIGHT * CAT_WIDTH, CAT_WIDTH, 1),
    )
    arg357_1 = _require_f32_tensor("arg357_1", inputs[1], (BRANCH0_CHANNELS,), (1,))
    convolution_71 = _require_f32_tensor(
        "convolution_71",
        inputs[2],
        (BATCH, BRANCH0_CHANNELS, HEIGHT, WIDTH),
        (BRANCH0_CHANNELS * HW, HW, WIDTH, 1),
    )
    arg358_1 = _require_f32_tensor("arg358_1", inputs[3], (BRANCH0_CHANNELS,), (1,))
    arg359_1 = _require_f32_tensor("arg359_1", inputs[4], (BRANCH0_CHANNELS,), (1,))
    arg360_1 = _require_f32_tensor("arg360_1", inputs[5], (BRANCH0_CHANNELS,), (1,))
    arg377_1 = _require_f32_tensor("arg377_1", inputs[6], (BRANCH1_CHANNELS,), (1,))
    convolution_75 = _require_f32_tensor(
        "convolution_75",
        inputs[7],
        (BATCH, BRANCH1_CHANNELS, HEIGHT, WIDTH),
        (BRANCH1_CHANNELS * HW, HW, WIDTH, 1),
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


def _torch_oracle(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
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
    values, offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        cat_7, [3, 3], [2, 2], [0, 0], [1, 1], False
    )
    branch0 = _branch_torch(convolution_71, arg357_1, arg358_1, arg359_1, arg360_1)
    branch1 = _branch_torch(convolution_75, arg377_1, arg378_1, arg379_1, arg380_1)
    cat = torch.cat([branch0, branch1, values], 1)
    avg = torch.ops.aten.avg_pool2d.default(cat, [3, 3], [1, 1], [1, 1])
    return offsets, avg


@oracle_impl(hardware="H100", shapes="(T([128, 768, 17, 17], f32), T([320], f32), T([128, 320, 8, 8], f32), T([320], f32), T([320], f32), T([320], f32), T([192], f32), T([128, 192, 8, 8], f32), T([192], f32), T([192], f32), T([192], f32))")
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
    checked = _validate_inputs(inputs)
    if not checked[0].is_cuda:
        return _torch_oracle(checked)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    cat_7 = checked[0]
    offsets = torch.empty_strided(
        (BATCH, CAT_CHANNELS, HEIGHT, WIDTH),
        (CAT_CHANNELS * HW, HW, WIDTH, 1),
        device=cat_7.device,
        dtype=torch.int8,
    )
    maxpool_values = torch.empty_strided(
        (BATCH, CAT_CHANNELS, HEIGHT, WIDTH),
        (CAT_CHANNELS * HW, HW, WIDTH, 1),
        device=cat_7.device,
        dtype=torch.float32,
    )
    avg = torch.empty_strided(
        (BATCH, AVG_CHANNELS, HEIGHT, WIDTH),
        (AVG_CHANNELS * HW, HW, WIDTH, 1),
        device=cat_7.device,
        dtype=torch.float32,
    )

    maxpool_grid = (BATCH, triton.cdiv(CAT_CHANNELS, BLOCK_C))
    _maxpool_values_offsets_kernel[maxpool_grid](
        cat_7,
        maxpool_values,
        offsets,
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
        num_warps=4,
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
        num_warps=4,
        num_stages=3,
    )

    pooled_grid = (BATCH, triton.cdiv(CAT_CHANNELS, BLOCK_C))
    _pooled_avgpool_kernel[pooled_grid](
        maxpool_values,
        avg,
        BLOCK_C_=BLOCK_C,
        BLOCK_HW_=BLOCK_HW,
        num_warps=4,
        num_stages=3,
    )
    return offsets, avg


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
