"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete NFNet avg-pool-backward SiLU-gradient channel reduction returned by Repro.forward, fusing the 2x2 avg_pool2d_backward producer, the captured `exp(-x) + 1` SiLU derivative chain, the spatial sum, and the sigmoid-gradient epilogue into one per-(N,C) Triton reduction before a small channel finalizer, whereas tuned Inductor already measures within the CUDAGraph floor for the same full scope; Inductor does not need a local scheduler, scatter-reduce, split-K, algebraic-elimination, or recompute-fusion fix for this repro because the dominant cost is the mandatory f32 scan of the three 28x28 inputs, the exact libdevice exp math, and the channel reduction; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader transcendental/reduction codegen work moves both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
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
    has_stochastic_ops,
)


N = 128
C = 512
POOL_H = 14
POOL_W = 14
POOL_HW = POOL_H * POOL_W
H = 28
W = 28
HW = H * W
SPATIAL_BLOCK = 1024
FINAL_BLOCK_N = 128

POOL_SHAPE = (N, C, POOL_H, POOL_W)
POOL_STRIDE = (C * POOL_HW, POOL_HW, POOL_W, 1)
FULL_SHAPE = (N, C, H, W)
FULL_STRIDE = (C * HW, HW, W, 1)
GATE_SHAPE = (N, C, 1, 1)
GATE_STRIDE = (C, 1, 1, 1)
OUT_SHAPE = (C,)
OUT_STRIDE = (1,)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _spatial_gate_partial_kernel(
        pool_grad_ptr,
        getitem_165_ptr,
        arg219_ptr,
        arg215_ptr,
        gate_ptr,
        partial_ptr,
        BLOCK_HW: tl.constexpr,
    ):
        c = tl.program_id(0)
        n = tl.program_id(1)
        hw = tl.arange(0, BLOCK_HW)
        mask = hw < 784

        h = hw // 28
        w = hw - h * 28
        pool_hw = (h // 2) * 14 + (w // 2)

        full_base = n * 401408 + c * 784
        pool_base = n * 100352 + c * 196
        full_offsets = full_base + hw
        pool_offsets = pool_base + pool_hw

        pool_grad = tl.load(pool_grad_ptr + pool_offsets, mask=mask, other=0.0).to(tl.float32) * 0.25
        getitem_165 = tl.load(getitem_165_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
        arg219 = tl.load(arg219_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
        arg215 = tl.load(arg215_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)

        add_tensor = getitem_165 + pool_grad
        mul_tensor = add_tensor * 0.9622504486493761

        neg_default = -arg219
        exp_default = libdevice.exp(neg_default)
        add_tensor_1 = exp_default + 1.0
        reciprocal_default = 1.0 / add_tensor_1
        mul_tensor_1 = reciprocal_default * 1.0
        mul_tensor_2 = mul_tensor * mul_tensor_1
        sub_tensor = 1.0 - mul_tensor_1
        mul_tensor_3 = arg219 * sub_tensor
        add_tensor_2 = mul_tensor_3 + 1.0
        mul_tensor_4 = mul_tensor_2 * add_tensor_2
        mul_tensor_5 = mul_tensor_4 * 0.2
        mul_tensor_6 = mul_tensor_5 * 2.0
        mul_tensor_7 = mul_tensor_6 * arg215
        spatial_sum = tl.sum(tl.where(mask, mul_tensor_7, 0.0), axis=0)

        gate = tl.load(gate_ptr + n * 512 + c).to(tl.float32)
        gate_sigmoid = tl.sigmoid(gate)
        sub_tensor_1 = 1.0 - gate_sigmoid
        mul_tensor_8 = gate_sigmoid * sub_tensor_1
        tl.store(partial_ptr + c * 128 + n, spatial_sum * mul_tensor_8)

    @triton.jit
    def _final_channel_sum_kernel(
        partial_ptr,
        out_ptr,
        BLOCK_N: tl.constexpr,
    ):
        c = tl.program_id(0)
        n = tl.arange(0, BLOCK_N)
        vals = tl.load(partial_ptr + c * 128 + n, mask=n < 128, other=0.0).to(tl.float32)
        tl.store(out_ptr + c, tl.sum(vals, axis=0))


def _torch_oracle(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    (
        getitem_168,
        arg220_1,
        getitem_165,
        arg219_1,
        arg215_1,
        arg218_1,
    ) = inputs
    avg_pool2d_backward_default = torch.ops.aten.avg_pool2d_backward.default(
        getitem_168,
        arg220_1,
        [2, 2],
        [2, 2],
        [0, 0],
        True,
        False,
        None,
    )
    add_tensor = torch.ops.aten.add.Tensor(getitem_165, avg_pool2d_backward_default)
    mul_tensor = torch.ops.aten.mul.Tensor(add_tensor, 0.9622504486493761)
    neg_default = torch.ops.aten.neg.default(arg219_1)
    exp_default = torch.ops.aten.exp.default(neg_default)
    add_tensor_1 = torch.ops.aten.add.Tensor(exp_default, 1)
    reciprocal_default = torch.ops.aten.reciprocal.default(add_tensor_1)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(reciprocal_default, 1)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_1)
    sub_tensor = torch.ops.aten.sub.Tensor(1, mul_tensor_1)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(arg219_1, sub_tensor)
    add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_3, 1)
    mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_2)
    mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.2)
    mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, 2.0)
    mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, arg215_1)
    sigmoid_default = torch.ops.aten.sigmoid.default(arg218_1)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2, 3], True)
    sub_tensor_1 = torch.ops.aten.sub.Tensor(1, sigmoid_default)
    mul_tensor_8 = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1)
    mul_tensor_9 = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_8)
    return torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 2, 3])


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


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")
    (
        getitem_168,
        arg220_1,
        getitem_165,
        arg219_1,
        arg215_1,
        arg218_1,
    ) = inputs

    getitem_168 = _require_f32_tensor("getitem_168", getitem_168, POOL_SHAPE, POOL_STRIDE)
    arg220_1 = _require_f32_tensor("arg220_1", arg220_1, FULL_SHAPE, FULL_STRIDE)
    getitem_165 = _require_f32_tensor("getitem_165", getitem_165, FULL_SHAPE, FULL_STRIDE)
    arg219_1 = _require_f32_tensor("arg219_1", arg219_1, FULL_SHAPE, FULL_STRIDE)
    arg215_1 = _require_f32_tensor("arg215_1", arg215_1, FULL_SHAPE, FULL_STRIDE)
    arg218_1 = _require_f32_tensor("arg218_1", arg218_1, GATE_SHAPE, GATE_STRIDE)

    devices = {x.device for x in (getitem_168, arg220_1, getitem_165, arg219_1, arg215_1, arg218_1)}
    if len(devices) != 1:
        raise ValueError("all tensor inputs must be on the same device")
    return getitem_168, arg220_1, getitem_165, arg219_1, arg215_1, arg218_1


@oracle_impl(hardware="H100", shapes="(T([128, 512, 14, 14], f32), T([128, 512, 28, 28], f32), T([128, 512, 28, 28], f32), T([128, 512, 28, 28], f32), T([128, 512, 28, 28], f32), T([128, 512, 1, 1], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation with the same output shape/dtype/stride."""
    (
        getitem_168,
        _arg220_1,
        getitem_165,
        arg219_1,
        arg215_1,
        arg218_1,
    ) = _validate_inputs(inputs)

    if getitem_168.device.type != "cuda":
        return _torch_oracle(inputs)
    if triton is None or libdevice is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    partial = torch.empty_strided((C, N), (N, 1), device=getitem_168.device, dtype=torch.float32)
    _spatial_gate_partial_kernel[(C, N)](
        getitem_168,
        getitem_165,
        arg219_1,
        arg215_1,
        arg218_1,
        partial,
        BLOCK_HW=SPATIAL_BLOCK,
        num_warps=8,
        num_stages=1,
    )

    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=getitem_168.device, dtype=torch.float32)
    _final_channel_sum_kernel[(C,)](
        partial,
        out,
        BLOCK_N=FINAL_BLOCK_N,
        num_warps=4,
        num_stages=1,
    )
    return out


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
