"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ViT patch/class-token positional-add LayerNorm scope in one shape-specialized Triton row kernel, including the convolution view/permute gather, class-token expand, virtual cat, positional add, fp32 var_mean(correction=0, keepdim=True) with eps=1e-6, affine epilogue, and final flattened contiguous output, whereas tuned Inductor already emits a comparable single fused row-reduction kernel and measures faster on the required harness; Inductor cannot materially improve this local repro through a narrower norm-template canonicalization change because the remaining work is the mandatory patch/class/position/affine reads, hidden-dim reduction, and output write rather than avoidable intermediate traffic; the fix is BANDWIDTH_BOUND: record this as an at-floor structural oracle and mark the measured oracle result not_true_floor unless a broader normalization codegen improvement beats the current compiled kernel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

EPS = 1.0e-6

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=3),
        ],
        key=["num_rows", "hidden", "tokens", "conv_stride_c", "conv_stride_patch"],
    )
    @triton.jit
    def _vit_class_token_layernorm_kernel(
        convolution_ptr,
        class_token_ptr,
        position_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        conv_stride_b: tl.constexpr,
        conv_stride_c: tl.constexpr,
        conv_stride_patch: tl.constexpr,
        class_stride_c: tl.constexpr,
        position_stride_t: tl.constexpr,
        position_stride_c: tl.constexpr,
        num_rows: tl.constexpr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        row_mask = rows < num_rows
        col_mask = cols < hidden
        mask = row_mask[:, None] & col_mask[None, :]

        batch = rows // tokens
        token = rows - batch * tokens
        is_class_token = token == 0
        patch = token - 1
        safe_patch = tl.maximum(patch, 0)

        class_values = tl.load(
            class_token_ptr + cols * class_stride_c,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        patch_values = tl.load(
            convolution_ptr
            + batch[:, None] * conv_stride_b
            + cols[None, :] * conv_stride_c
            + safe_patch[:, None] * conv_stride_patch,
            mask=mask & (token[:, None] != 0),
            other=0.0,
        ).to(tl.float32)
        position_values = tl.load(
            position_ptr
            + token[:, None] * position_stride_t
            + cols[None, :] * position_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        values = tl.where(is_class_token[:, None], class_values[None, :], patch_values)
        values = tl.where(mask, values + position_values, 0.0)
        mean = tl.sum(values, axis=1) / hidden
        sum_x2 = tl.sum(values * values, axis=1)
        variance = tl.maximum(sum_x2 / hidden - mean * mean, 0.0)
        centered = values - mean[:, None]
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        output_values = centered * invstd[:, None] * weight[None, :] + bias[None, :]

        tl.store(
            output_ptr + rows[:, None] * hidden + cols[None, :],
            output_values,
            mask=mask,
        )


def _shape_tuple(value):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def _validate_inputs(inputs):
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        convolution,
        class_token,
        position,
        weight,
        bias,
        conv_view_shape,
        expand_shape,
        output_shape,
    ) = inputs

    tensor_inputs = (convolution, class_token, position, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    for index, value in enumerate(tensor_inputs):
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")

    if convolution.ndim != 4:
        raise ValueError(f"convolution must be rank 4, got shape {tuple(convolution.shape)}")
    batch, hidden, height, width = (int(dim) for dim in convolution.shape)
    patches = height * width
    tokens = patches + 1
    rows = batch * tokens

    expected_tensor_shapes = (
        (batch, hidden, height, width),
        (1, 1, hidden),
        (1, tokens, hidden),
        (hidden,),
        (hidden,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_tensor_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")

    conv_view_shape_tuple = _shape_tuple(conv_view_shape)
    expand_shape_tuple = _shape_tuple(expand_shape)
    output_shape_tuple = _shape_tuple(output_shape)
    if conv_view_shape_tuple != (batch, hidden, patches):
        raise ValueError(f"unexpected convolution view shape parameter: {conv_view_shape!r}")
    if expand_shape_tuple not in ((batch, 1, hidden), (batch, -1, -1)):
        raise ValueError(f"unexpected expand shape parameter: {expand_shape!r}")
    if output_shape_tuple != (rows, hidden):
        raise ValueError(f"unexpected output shape parameter: {output_shape!r}")
    if convolution.stride(2) != width * convolution.stride(3):
        raise ValueError(f"convolution H/W dims cannot be viewed as flat patches: stride={convolution.stride()}")

    devices = {value.device for value in tensor_inputs}
    if len(devices) != 1:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return convolution, class_token, position, weight, bias, output_shape_tuple, tokens, width


@oracle_impl(hardware="H100", shapes="(T([128, 192, 14, 14], f32), T([1, 1, 192], f32), T([1, 197, 192], f32), T([192], f32), T([192], f32), S([128, 192, 196]), S([128, -1, -1]), S([25216, 192]))")
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
    if triton is None:
        raise RuntimeError("Triton is required for oracle_vit_class_token_layernorm.py")

    convolution, class_token, position, weight, bias, output_shape, tokens, _width = _validate_inputs(inputs)
    rows, hidden = output_shape
    output = torch.empty_strided(
        output_shape,
        (hidden, 1),
        device=convolution.device,
        dtype=torch.float32,
    )
    block_n = _next_power_of_2(hidden)
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]),)
    _vit_class_token_layernorm_kernel[grid](
        convolution,
        class_token,
        position,
        weight,
        bias,
        output,
        convolution.stride(0),
        convolution.stride(1),
        convolution.stride(3),
        class_token.stride(2),
        position.stride(1),
        position.stride(2),
        num_rows=rows,
        hidden=hidden,
        tokens=tokens,
        eps=EPS,
        BLOCK_N=block_n,
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
