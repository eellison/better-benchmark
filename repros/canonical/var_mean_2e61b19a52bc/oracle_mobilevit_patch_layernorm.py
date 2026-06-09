"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileViT patch-canonicalization plus affine LayerNorm scope directly from the channels-last `[128,240,8,8]` source in one row-wise Triton kernel, including the three clone/reshape/permute layout rewrites, fp32 population var_mean over 240 channels, eps=1e-5 rsqrt, affine scale/bias, and final contiguous `[8192,240]` output, whereas tuned Inductor already reaches the same practical launch and memory-traffic envelope for this fixed 2x2 patch-unfold LayerNorm case; Inductor cannot materially improve this repro through a narrower norm-template canonicalization change because the remaining work is the required activation/affine reads, row reduction, rsqrt, output store, and launch overhead rather than avoidable intermediate materialization; the fix is BANDWIDTH_BOUND: record this as an at-floor MobileViT patch LayerNorm case unless broader normalization, layout-copy, or launch-overhead improvements move both paths."""
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


EPS = 1.0e-5
PATCH_H = 2
PATCH_W = 2
PATCH_LANES = PATCH_H * PATCH_W
ROW_BLOCK = 4


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK_META": 1}, num_warps=1, num_stages=3),
            triton.Config({"ROW_BLOCK_META": 2}, num_warps=1, num_stages=3),
            triton.Config({"ROW_BLOCK_META": 4}, num_warps=1, num_stages=3),
            triton.Config({"ROW_BLOCK_META": 4}, num_warps=2, num_stages=3),
            triton.Config({"ROW_BLOCK_META": 8}, num_warps=2, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _patch_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        patches: tl.constexpr,
        num_patch_w: tl.constexpr,
        patch_h: tl.constexpr,
        patch_w: tl.constexpr,
        patch_lanes: tl.constexpr,
        stride_n: tl.constexpr,
        stride_c: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        ROW_BLOCK_META: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK_META + tl.arange(0, ROW_BLOCK_META)
        cols = tl.arange(0, block_h)
        row_mask = row_ids < total_rows
        col_mask = cols < hidden
        mask = row_mask[:, None] & col_mask[None, :]

        patch = row_ids % patches
        row_group = row_ids // patches
        lane = row_group % patch_lanes
        batch = row_group // patch_lanes
        h = (patch // num_patch_w) * patch_h + (lane // patch_w)
        w = (patch % num_patch_w) * patch_w + (lane % patch_w)

        input_offsets = (
            batch[:, None] * stride_n
            + cols[None, :] * stride_c
            + h[:, None] * stride_h
            + w[:, None] * stride_w
        )
        output_offsets = row_ids[:, None] * hidden + cols[None, :]

        x = tl.load(
            x_ptr + input_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
        invstd = tl.rsqrt(variance + eps)

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
        y = centered * invstd * weight[None, :] + bias[None, :]
        tl.store(out_ptr + output_offsets, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int],
    tuple[int, int],
]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    x, weight, bias, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("the first three repro inputs must be tensors")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    shape2_tuple = _shape_tuple(shape2)
    shape3_tuple = _shape_tuple(shape3)
    if len(shape0_tuple) != 4 or len(shape1_tuple) != 4 or len(shape2_tuple) != 3 or len(shape3_tuple) != 2:
        raise ValueError(
            f"unexpected shape parameters: {shape0_tuple}, {shape1_tuple}, "
            f"{shape2_tuple}, {shape3_tuple}"
        )

    batch, hidden, height, width = tuple(int(dim) for dim in x.shape)
    if height % PATCH_H != 0 or width % PATCH_W != 0:
        raise ValueError(f"expected even spatial size, got {(height, width)}")

    num_patch_h = height // PATCH_H
    num_patch_w = width // PATCH_W
    patches = num_patch_h * num_patch_w
    total_rows = batch * PATCH_LANES * patches

    expected_shape0 = (batch * hidden * num_patch_h, PATCH_H, num_patch_w, PATCH_W)
    expected_shape1 = (batch, hidden, patches, PATCH_LANES)
    expected_shape2 = (batch * PATCH_LANES, patches, hidden)
    expected_shape3 = (total_rows, hidden)
    if shape0_tuple != expected_shape0:
        raise ValueError(f"shape0 {shape0_tuple} != {expected_shape0}")
    if shape1_tuple != expected_shape1:
        raise ValueError(f"shape1 {shape1_tuple} != {expected_shape1}")
    if shape2_tuple != expected_shape2:
        raise ValueError(f"shape2 {shape2_tuple} != {expected_shape2}")
    if shape3_tuple != expected_shape3:
        raise ValueError(f"shape3 {shape3_tuple} != {expected_shape3}")

    for index, value in enumerate((x, weight, bias)):
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if value.device != x.device:
            raise ValueError(f"input {index} device {value.device} != {x.device}")

    if tuple(weight.shape) != (hidden,) or tuple(bias.shape) != (hidden,):
        raise ValueError(
            f"affine shapes weight={tuple(weight.shape)} bias={tuple(bias.shape)} "
            f"must both be ({hidden},)"
        )
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("affine weight and bias must be contiguous")

    return x, weight, bias, shape0_tuple, shape1_tuple, shape2_tuple, shape3_tuple


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    x, weight, bias, shape0, shape1, shape2, shape3 = _validate_inputs(inputs)
    clone_default = torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)
    reshape_default = torch.ops.aten.reshape.default(clone_default, shape0)
    permute_default = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3])
    clone_default_1 = torch.ops.aten.clone.default(
        permute_default,
        memory_format=torch.contiguous_format,
    )
    reshape_default_1 = torch.ops.aten.reshape.default(clone_default_1, shape1)
    permute_default_1 = torch.ops.aten.permute.default(reshape_default_1, [0, 3, 2, 1])
    clone_default_2 = torch.ops.aten.clone.default(
        permute_default_1,
        memory_format=torch.contiguous_format,
    )
    reshape_default_2 = torch.ops.aten.reshape.default(clone_default_2, shape2)
    var, mean = torch.ops.aten.var_mean.correction(
        reshape_default_2,
        [2],
        correction=0,
        keepdim=True,
    )
    normalized = torch.ops.aten.mul.Tensor(
        torch.ops.aten.sub.Tensor(reshape_default_2, mean),
        torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var, EPS)),
    )
    affine = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(normalized, weight),
        bias,
    )
    return torch.ops.aten.reshape.default(affine, shape3)


@oracle_impl(hardware="H100", shapes="(T([128, 240, 8, 8], f32, stride=(15360, 1, 1920, 240)), T([240], f32), T([240], f32), S([122880, 2, 4, 2]), S([128, 240, 16, 4]), S([512, 16, 240]), S([8192, 240]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward patch-unfold plus affine LayerNorm scope."""
    x, weight, bias, _shape0, _shape1, _shape2, shape3 = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_full_scope(inputs)

    batch, hidden, height, width = tuple(int(dim) for dim in x.shape)
    patches = (height // PATCH_H) * (width // PATCH_W)
    output = torch.empty_strided(
        shape3,
        (hidden, 1),
        device=x.device,
        dtype=torch.float32,
    )
    block_h = triton.next_power_of_2(hidden)
    grid = lambda meta: (triton.cdiv(shape3[0], meta["ROW_BLOCK_META"]),)
    _patch_layernorm_kernel[grid](
        x,
        weight,
        bias,
        output,
        total_rows=shape3[0],
        hidden=hidden,
        patches=patches,
        num_patch_w=width // PATCH_W,
        patch_h=PATCH_H,
        patch_w=PATCH_W,
        patch_lanes=PATCH_LANES,
        stride_n=x.stride(0),
        stride_c=x.stride(1),
        stride_h=x.stride(2),
        stride_w=x.stride(3),
        eps=EPS,
        block_h=block_h,
    )
    del batch
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
