"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ResNeSt split-attention softmax-gradient block returned by Repro.forward, including the broadcasted input expansion, affine scale/bias, ReLU, spatial per-group weighted sums, stable two-way softmax, FMA-style softmax-backward epilogue, layout views/permutes, and final channel reduction in two Triton kernels, whereas Inductor's compiled full-scope lowering is already faster under the required CUDAGraph harness; Inductor cannot be assigned a material local scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, or new-pattern gap from this measurement because the mandatory activation reads, spatial reductions, and small grouped softmax batch reduction dominate the full scope; the fix is BANDWIDTH_BOUND: record this diagnostic oracle as not a true performance floor unless a faster full-scope implementation beats tuned Inductor on the same harness."""
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


BATCH = 32
GROUPS = 2
CHANNELS = 64
TOTAL_CHANNELS = GROUPS * CHANNELS
HEIGHT = 56
WIDTH = 56
HW = HEIGHT * WIDTH
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    *,
    strides: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if strides is not None and tuple(value.stride()) != strides:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {strides}")
    return value


def _expect_shape_param(name: str, value: Any, expected: tuple[int, ...]) -> None:
    actual = _shape_tuple(value)
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {actual}")


def _validate_inputs(inputs):
    if len(inputs) != 14:
        raise ValueError(f"expected 14 inputs, got {len(inputs)}")

    (
        getitem_57,
        arg73_1,
        arg74_1,
        arg75_1,
        arg11_1,
        arg12_1,
        arg80_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    ) = inputs

    _expect_tensor(
        "getitem_57",
        getitem_57,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        strides=(CHANNELS * HW, HW, WIDTH, 1),
    )
    _expect_tensor(
        "arg73_1",
        arg73_1,
        (BATCH, TOTAL_CHANNELS, HEIGHT, WIDTH),
        strides=(TOTAL_CHANNELS * HW, HW, WIDTH, 1),
    )
    _expect_tensor("arg74_1", arg74_1, (1, TOTAL_CHANNELS, 1, 1), strides=(TOTAL_CHANNELS, 1, 1, 1))
    _expect_tensor("arg75_1", arg75_1, (1, TOTAL_CHANNELS, 1, 1), strides=(TOTAL_CHANNELS, 1, 1, 1))
    _expect_tensor("arg11_1", arg11_1, (TOTAL_CHANNELS,), strides=(1,))
    _expect_tensor("arg12_1", arg12_1, (TOTAL_CHANNELS,), strides=(1,))
    _expect_tensor("arg80_1", arg80_1, (BATCH, TOTAL_CHANNELS, 1, 1), strides=(TOTAL_CHANNELS, 1, 1, 1))

    _expect_shape_param("_shape_param_0", _shape_param_0, (BATCH, GROUPS, CHANNELS, HEIGHT, WIDTH))
    _expect_shape_param("_shape_param_1", _shape_param_1, (BATCH, GROUPS, CHANNELS, HEIGHT, WIDTH))
    _expect_shape_param("_shape_param_2", _shape_param_2, (BATCH, 1, GROUPS, -1))
    _expect_shape_param("_shape_param_3", _shape_param_3, (BATCH, TOTAL_CHANNELS, 1, 1))
    _expect_shape_param("_shape_param_4", _shape_param_4, (BATCH, TOTAL_CHANNELS))
    _expect_shape_param("_shape_param_5", _shape_param_5, (BATCH, GROUPS, 1, CHANNELS))
    _expect_shape_param("_shape_param_6", _shape_param_6, (BATCH, TOTAL_CHANNELS, 1, 1))

    return (
        getitem_57,
        arg73_1,
        arg74_1,
        arg75_1,
        arg11_1,
        arg12_1,
        arg80_1,
    )


def _all_cuda_contiguous(tensors: tuple[torch.Tensor, ...]) -> bool:
    return all(t.is_cuda and t.is_contiguous() for t in tensors)


def _aten_fallback(inputs):
    (
        getitem_57,
        arg73_1,
        arg74_1,
        arg75_1,
        arg11_1,
        arg12_1,
        arg80_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    ) = inputs

    unsqueeze_default = torch.ops.aten.unsqueeze.default(getitem_57, 1)
    expand_default = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0)
    sub_tensor = torch.ops.aten.sub.Tensor(arg73_1, arg74_1)
    mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, arg75_1)
    unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(arg11_1, -1)
    unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, -1)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_2)
    unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(arg12_1, -1)
    unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, -1)
    add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_4)
    relu_default = torch.ops.aten.relu.default(add_tensor)
    view_default = torch.ops.aten.view.default(relu_default, _shape_param_1)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(expand_default, view_default)
    view_default_1 = torch.ops.aten.view.default(arg80_1, _shape_param_2)
    permute_default = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3])
    amax_default = torch.ops.aten.amax.default(permute_default, [1], True)
    sub_tensor_1 = torch.ops.aten.sub.Tensor(permute_default, amax_default)
    exp_default = torch.ops.aten.exp.default(sub_tensor_1)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True)
    div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list)
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3, 4], True)
    view_default_2 = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_3)
    view_default_3 = torch.ops.aten.view.default(view_default_2, _shape_param_4)
    view_default_4 = torch.ops.aten.view.default(view_default_3, _shape_param_5)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(view_default_4, div_tensor)
    sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [1], True)
    neg_default = torch.ops.aten.neg.default(div_tensor)
    fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list_2, mul_tensor_3)
    permute_default_1 = torch.ops.aten.permute.default(fma_default, [0, 2, 1, 3])
    view_default_5 = torch.ops.aten.view.default(permute_default_1, _shape_param_6)
    return torch.ops.aten.sum.dim_IntList(view_default_5, [0, 2, 3])


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _spatial_weighted_sum_kernel(
        getitem_ptr,
        activation_ptr,
        mean_ptr,
        invstd_scale_ptr,
        weight_ptr,
        bias_ptr,
        tmp_ptr,
        hw: tl.constexpr,
        channels: tl.constexpr,
        total_channels: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        batch = tl.program_id(0)
        channel_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.arange(0, BLOCK_HW)
        channel_mask = channel_offsets < channels
        hw_mask = hw_offsets < hw
        mask = channel_mask[:, None] & hw_mask[None, :]

        getitem_offsets = (batch * channels + channel_offsets[:, None]) * hw + hw_offsets[None, :]
        x = tl.load(getitem_ptr + getitem_offsets, mask=mask, other=0.0).to(tl.float32)

        channel0 = channel_offsets
        channel1 = channel_offsets + channels
        act_offsets0 = (batch * total_channels + channel0[:, None]) * hw + hw_offsets[None, :]
        act_offsets1 = (batch * total_channels + channel1[:, None]) * hw + hw_offsets[None, :]

        act0 = tl.load(activation_ptr + act_offsets0, mask=mask, other=0.0).to(tl.float32)
        act1 = tl.load(activation_ptr + act_offsets1, mask=mask, other=0.0).to(tl.float32)

        mean0 = tl.load(mean_ptr + channel0, mask=channel_mask, other=0.0).to(tl.float32)
        mean1 = tl.load(mean_ptr + channel1, mask=channel_mask, other=0.0).to(tl.float32)
        scale0 = tl.load(invstd_scale_ptr + channel0, mask=channel_mask, other=0.0).to(tl.float32)
        scale1 = tl.load(invstd_scale_ptr + channel1, mask=channel_mask, other=0.0).to(tl.float32)
        weight0 = tl.load(weight_ptr + channel0, mask=channel_mask, other=0.0).to(tl.float32)
        weight1 = tl.load(weight_ptr + channel1, mask=channel_mask, other=0.0).to(tl.float32)
        bias0 = tl.load(bias_ptr + channel0, mask=channel_mask, other=0.0).to(tl.float32)
        bias1 = tl.load(bias_ptr + channel1, mask=channel_mask, other=0.0).to(tl.float32)

        relu0 = tl.maximum(((act0 - mean0[:, None]) * scale0[:, None]) * weight0[:, None] + bias0[:, None], 0.0)
        relu1 = tl.maximum(((act1 - mean1[:, None]) * scale1[:, None]) * weight1[:, None] + bias1[:, None], 0.0)
        sum0 = tl.sum(x * relu0, axis=1)
        sum1 = tl.sum(x * relu1, axis=1)

        tmp_base = batch * total_channels + channel_offsets
        tl.store(tmp_ptr + tmp_base, sum0, mask=channel_mask)
        tl.store(tmp_ptr + tmp_base + channels, sum1, mask=channel_mask)

    @triton.jit
    def _softmax_backward_reduce_kernel(
        logits_ptr,
        tmp_ptr,
        output_ptr,
        batch_size: tl.constexpr,
        channels: tl.constexpr,
        total_channels: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        channel_offsets = tl.arange(0, BLOCK_C)
        batch_offsets = tl.arange(0, BLOCK_B)
        channel_mask = channel_offsets < channels
        batch_mask = batch_offsets < batch_size
        mask = channel_mask[:, None] & batch_mask[None, :]

        tmp_base = batch_offsets[None, :] * total_channels + channel_offsets[:, None]
        t0 = tl.load(tmp_ptr + tmp_base, mask=mask, other=0.0).to(tl.float32)
        t1 = tl.load(tmp_ptr + tmp_base + channels, mask=mask, other=0.0).to(tl.float32)

        logits_base = batch_offsets[None, :] * total_channels + channel_offsets[:, None]
        logit0 = tl.load(logits_ptr + logits_base, mask=mask, other=0.0).to(tl.float32)
        logit1 = tl.load(logits_ptr + logits_base + channels, mask=mask, other=0.0).to(tl.float32)

        row_max = tl.maximum(logit0, logit1)
        exp0 = tl.exp(logit0 - row_max)
        exp1 = tl.exp(logit1 - row_max)
        denom = exp0 + exp1
        prob0 = exp0 / denom
        prob1 = exp1 / denom

        weighted = prob0 * t0 + prob1 * t1
        grad0 = (-prob0 * weighted) + (prob0 * t0)
        grad1 = (-prob1 * weighted) + (prob1 * t1)
        out0 = tl.sum(tl.where(batch_mask[None, :], grad0, 0.0), axis=1)
        out1 = tl.sum(tl.where(batch_mask[None, :], grad1, 0.0), axis=1)

        tl.store(output_ptr + channel_offsets, out0, mask=channel_mask)
        tl.store(output_ptr + channels + channel_offsets, out1, mask=channel_mask)


@oracle_impl(hardware="H100", shapes="(T([32, 64, 56, 56], f32), T([32, 128, 56, 56], f32), T([1, 128, 1, 1], f32), T([1, 128, 1, 1], f32), T([128], f32), T([128], f32), T([32, 128, 1, 1], f32), S([32, 2, 64, 56, 56]), S([32, 2, 64, 56, 56]), S([32, 1, 2, -1]), S([32, 128, 1, 1]), S([32, 128]), S([32, 2, 1, 64]), S([32, 128, 1, 1]))")
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
    tensors = _validate_inputs(inputs)
    (
        getitem_57,
        arg73_1,
        arg74_1,
        arg75_1,
        arg11_1,
        arg12_1,
        arg80_1,
    ) = tensors

    if triton is None or not _all_cuda_contiguous(tensors):
        return _aten_fallback(inputs)

    tmp = torch.empty((BATCH, GROUPS, CHANNELS), device=getitem_57.device, dtype=torch.float32)
    output = torch.empty((TOTAL_CHANNELS,), device=getitem_57.device, dtype=torch.float32)

    block_c = 2
    _spatial_weighted_sum_kernel[(BATCH, triton.cdiv(CHANNELS, block_c))](
        getitem_57,
        arg73_1,
        arg74_1,
        arg75_1,
        arg11_1,
        arg12_1,
        tmp,
        hw=HW,
        channels=CHANNELS,
        total_channels=TOTAL_CHANNELS,
        BLOCK_C=block_c,
        BLOCK_HW=triton.next_power_of_2(HW),
        num_warps=8,
    )
    _softmax_backward_reduce_kernel[(1,)](
        arg80_1,
        tmp,
        output,
        batch_size=BATCH,
        channels=CHANNELS,
        total_channels=TOTAL_CHANNELS,
        BLOCK_B=triton.next_power_of_2(BATCH),
        BLOCK_C=triton.next_power_of_2(CHANNELS),
        num_warps=4,
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
