"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete RegNet BN-affine/ReLU gated-gradient channel vector with the same spatial f32 reduction, sigmoid-derivative epilogue, and batch/channel f32 reduction structure as the generated Inductor kernels, whereas tuned Inductor already reaches the same CUDAGraph-measured floor for these two generic reductions; Inductor cannot materially improve this repro through a local fusion because the dominant work is the mandatory f32 scan of the two `[32, 224, 56, 56]` inputs plus affine/ReLU math, while the `[32, 224]` intermediate and second reduction are tiny; the fix is BANDWIDTH_BOUND: record this as at floor unless broader reduction memory/codegen work moves both implementations."""
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
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None
    triton_helpers = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    get_shape_key,
    has_stochastic_ops,
)


N = 32
C = 224
H = 56
W = 56
HW = H * W
INPUT_SHAPE = (N, C, H, W)
INPUT_STRIDE = (C * HW, HW, W, 1)
CHANNEL_BROADCAST_SHAPE = (1, C, 1, 1)
CHANNEL_BROADCAST_STRIDE = (C, 1, 1, 1)
SIGMOID_SHAPE = (N, C, 1, 1)
SIGMOID_STRIDE = (C, 1, 1, 1)
VECTOR_SHAPE = (C,)
VECTOR_STRIDE = (1,)
SPATIAL_XBLOCK = 1
SPATIAL_RBLOCK = 4096
FINAL_XBLOCK = 256
FINAL_RBLOCK = 32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _spatial_sum_kernel(
        getitem_ptr,
        arg191_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        partial_ptr,
        XNUMEL_: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        XBLOCK_: tl.constexpr,
        RBLOCK_: tl.constexpr,
    ):
        xindex = tl.program_id(0) * XBLOCK_ + tl.arange(0, XBLOCK_)[:, None]
        xmask = xindex < XNUMEL_
        rbase = tl.arange(0, RBLOCK_)[None, :]
        channel = xindex % C_

        mean = tl.load(mean_ptr + channel, mask=xmask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        invstd = tl.load(invstd_ptr + channel, mask=xmask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=xmask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=xmask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        acc = tl.zeros((XBLOCK_, RBLOCK_), dtype=tl.float32)

        for r_offset in tl.range(0, HW_, RBLOCK_):
            rindex = r_offset + rbase
            rmask = rindex < HW_
            mask = rmask & xmask
            row_offset = xindex * HW_ + rindex

            getitem = tl.load(
                getitem_ptr + row_offset,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            x = tl.load(
                arg191_ptr + row_offset,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            centered = x - mean
            normalized = centered * invstd
            scaled = normalized * weight
            shifted = scaled + bias
            relu = triton_helpers.maximum(tl.full((1, 1), 0, tl.int32), shifted)
            spatial_terms = getitem * relu
            acc = tl.where(mask, acc + spatial_terms, acc)

        spatial_sum = tl.sum(acc, axis=1)[:, None]
        tl.store(partial_ptr + xindex, spatial_sum, mask=xmask)

    @triton.jit
    def _final_sigmoid_sum_kernel(
        partial_ptr,
        sigmoid_input_ptr,
        out_ptr,
        C_: tl.constexpr,
        XBLOCK_: tl.constexpr,
        RBLOCK_: tl.constexpr,
    ):
        xindex = tl.program_id(0) * XBLOCK_ + tl.arange(0, XBLOCK_)[:, None]
        xmask = xindex < C_
        rindex = tl.arange(0, RBLOCK_)[None, :]

        offsets = xindex + C_ * rindex
        partial = tl.load(partial_ptr + offsets, mask=xmask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        sigmoid_input = tl.load(
            sigmoid_input_ptr + offsets,
            mask=xmask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        sigmoid = tl.sigmoid(sigmoid_input)
        sigmoid_deriv = sigmoid * (1.0 - sigmoid)
        contribution = partial * sigmoid_deriv
        reduced = tl.sum(tl.where(xmask, contribution, 0.0), axis=1)[:, None].to(tl.float32)
        tl.store(out_ptr + xindex, reduced, mask=xmask)


def _torch_oracle(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    (
        arg191_1,
        arg192_1,
        arg193_1,
        arg6_1,
        arg7_1,
        getitem_282,
        arg196_1,
    ) = inputs
    sub_tensor = torch.ops.aten.sub.Tensor(arg191_1, arg192_1)
    mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, arg193_1)
    unsqueeze_default = torch.ops.aten.unsqueeze.default(arg6_1, -1)
    unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1)
    unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(arg7_1, -1)
    unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1)
    add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3)
    relu_default = torch.ops.aten.relu.default(add_tensor)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(getitem_282, relu_default)
    sigmoid_default = torch.ops.aten.sigmoid.default(arg196_1)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True)
    sub_tensor_1 = torch.ops.aten.sub.Tensor(1, sigmoid_default)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1)
    mul_tensor_4 = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_3)
    return torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3])


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")
    (
        arg191_1,
        arg192_1,
        arg193_1,
        arg6_1,
        arg7_1,
        getitem_282,
        arg196_1,
    ) = inputs

    arg191_1 = _require_f32_tensor("arg191_1", arg191_1, INPUT_SHAPE, INPUT_STRIDE)
    arg192_1 = _require_f32_tensor(
        "arg192_1",
        arg192_1,
        CHANNEL_BROADCAST_SHAPE,
        CHANNEL_BROADCAST_STRIDE,
    )
    arg193_1 = _require_f32_tensor(
        "arg193_1",
        arg193_1,
        CHANNEL_BROADCAST_SHAPE,
        CHANNEL_BROADCAST_STRIDE,
    )
    arg6_1 = _require_f32_tensor("arg6_1", arg6_1, VECTOR_SHAPE, VECTOR_STRIDE)
    arg7_1 = _require_f32_tensor("arg7_1", arg7_1, VECTOR_SHAPE, VECTOR_STRIDE)
    getitem_282 = _require_f32_tensor("getitem_282", getitem_282, INPUT_SHAPE, INPUT_STRIDE)
    arg196_1 = _require_f32_tensor("arg196_1", arg196_1, SIGMOID_SHAPE, SIGMOID_STRIDE)

    devices = {
        arg191_1.device,
        arg192_1.device,
        arg193_1.device,
        arg6_1.device,
        arg7_1.device,
        getitem_282.device,
        arg196_1.device,
    }
    if len(devices) != 1:
        raise ValueError("all tensor inputs must be on the same device")
    return arg191_1, arg192_1, arg193_1, arg6_1, arg7_1, getitem_282, arg196_1


@oracle_impl(hardware="H100", shapes="(T([32, 224, 56, 56], f32), T([1, 224, 1, 1], f32), T([1, 224, 1, 1], f32), T([224], f32), T([224], f32), T([32, 224, 56, 56], f32), T([32, 224, 1, 1], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation with the same output shape/dtype/stride."""
    (
        arg191_1,
        arg192_1,
        arg193_1,
        arg6_1,
        arg7_1,
        getitem_282,
        arg196_1,
    ) = _validate_inputs(inputs)

    if arg191_1.device.type != "cuda":
        return _torch_oracle(inputs)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    partial = torch.empty_strided((N, C), (C, 1), device=arg191_1.device, dtype=torch.float32)
    _spatial_sum_kernel[(triton.cdiv(N * C, SPATIAL_XBLOCK),)](
        getitem_282,
        arg191_1,
        arg192_1,
        arg193_1,
        arg6_1,
        arg7_1,
        partial,
        XNUMEL_=N * C,
        C_=C,
        HW_=HW,
        XBLOCK_=SPATIAL_XBLOCK,
        RBLOCK_=SPATIAL_RBLOCK,
        num_warps=8,
        num_stages=1,
    )
    out = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=arg191_1.device, dtype=torch.float32)
    _final_sigmoid_sum_kernel[(triton.cdiv(C, FINAL_XBLOCK),)](
        partial,
        arg196_1,
        out,
        C_=C,
        XBLOCK_=FINAL_XBLOCK,
        RBLOCK_=FINAL_RBLOCK,
        num_warps=8,
        num_stages=1,
    )
    return out


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
