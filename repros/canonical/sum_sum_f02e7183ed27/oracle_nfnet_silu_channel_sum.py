"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete NFNet SiLU-gradient reduction returned by Repro.forward, fusing the captured `exp(-x) + 1` reciprocal chain, residual add, 0.2/2.0 scaling, spatial f32 sum, exact sigmoid-derivative gate, and final channel sum with one per-(N,C) Triton reduction plus a channel finalizer, whereas tuned Inductor already lowers the same full scope near the memory/transcendental floor; Inductor cannot materially improve this repro with scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the dominant work is the mandatory scan of four dense f32 activation tensors plus natural exp math and channel reduction; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader libdevice-exp/reduction codegen improvements move both implementations."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


N = 128
C = 1536
H = 14
W = 14
HW = H * W
SPATIAL_BLOCK = 256
REDUCE_BLOCK_C = 4
FINAL_BLOCK_N = 128

NCHW_SHAPE = (N, C, H, W)
NCHW_STRIDE = (C * HW, HW, W, 1)
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
        getitem_129_ptr,
        arg262_ptr,
        add_62_ptr,
        arg258_ptr,
        arg261_ptr,
        partial_ptr,
        BLOCK_HW: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        n = tl.program_id(1)
        c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw = tl.arange(0, BLOCK_HW)
        c_mask = c < 1536
        hw_mask = hw < 196
        mask = c_mask[:, None] & hw_mask[None, :]
        offsets = n * 301056 + c[:, None] * 196 + hw[None, :]

        getitem_129 = tl.load(getitem_129_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        arg262 = tl.load(arg262_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add_62 = tl.load(add_62_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        arg258 = tl.load(arg258_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        mul_tensor = getitem_129 * 0.9622504486493761
        neg_default = -arg262
        exp_default = libdevice.exp(neg_default)
        add_tensor = exp_default + 1.0
        reciprocal_default = 1.0 / add_tensor
        mul_tensor_1 = reciprocal_default * 1.0
        mul_tensor_2 = mul_tensor * mul_tensor_1
        sub_tensor = 1.0 - mul_tensor_1
        mul_tensor_3 = arg262 * sub_tensor
        add_tensor_1 = mul_tensor_3 + 1.0
        mul_tensor_4 = mul_tensor_2 * add_tensor_1
        add_tensor_2 = add_62 + mul_tensor_4
        mul_tensor_5 = add_tensor_2 * 0.2
        mul_tensor_6 = mul_tensor_5 * 2.0
        mul_tensor_7 = mul_tensor_6 * arg258
        spatial_sum = tl.sum(tl.where(hw_mask[None, :], mul_tensor_7, 0.0), axis=1)

        gate = tl.load(arg261_ptr + n * 1536 + c, mask=c_mask, other=0.0).to(tl.float32)
        gate_sigmoid = tl.sigmoid(gate)
        sub_tensor_1 = 1.0 - gate_sigmoid
        mul_tensor_8 = gate_sigmoid * sub_tensor_1
        mul_tensor_9 = spatial_sum * mul_tensor_8
        tl.store(partial_ptr + n * 1536 + c, mul_tensor_9, mask=c_mask)

    @triton.jit
    def _final_channel_sum_kernel(
        partial_ptr,
        out_ptr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        n = tl.arange(0, BLOCK_N)
        c_mask = c < 1536
        n_mask = n < 128
        vals = tl.load(
            partial_ptr + n[:, None] * 1536 + c[None, :],
            mask=n_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        total = tl.sum(vals, axis=0)
        tl.store(out_ptr + c, total, mask=c_mask)


def _torch_oracle(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    (
        getitem_129,
        arg262_1,
        add_62,
        arg258_1,
        arg261_1,
    ) = inputs
    mul_tensor = torch.ops.aten.mul.Tensor(getitem_129, 0.9622504486493761)
    neg_default = torch.ops.aten.neg.default(arg262_1)
    exp_default = torch.ops.aten.exp.default(neg_default)
    add_tensor = torch.ops.aten.add.Tensor(exp_default, 1)
    reciprocal_default = torch.ops.aten.reciprocal.default(add_tensor)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(reciprocal_default, 1)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_1)
    sub_tensor = torch.ops.aten.sub.Tensor(1, mul_tensor_1)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(arg262_1, sub_tensor)
    add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_3, 1)
    mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_1)
    add_tensor_2 = torch.ops.aten.add.Tensor(add_62, mul_tensor_4)
    mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_2, 0.2)
    mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, 2.0)
    mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, arg258_1)
    sigmoid_default = torch.ops.aten.sigmoid.default(arg261_1)
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
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")
    (
        getitem_129,
        arg262_1,
        add_62,
        arg258_1,
        arg261_1,
    ) = inputs

    getitem_129 = _require_f32_tensor("getitem_129", getitem_129, NCHW_SHAPE, NCHW_STRIDE)
    arg262_1 = _require_f32_tensor("arg262_1", arg262_1, NCHW_SHAPE, NCHW_STRIDE)
    add_62 = _require_f32_tensor("add_62", add_62, NCHW_SHAPE, NCHW_STRIDE)
    arg258_1 = _require_f32_tensor("arg258_1", arg258_1, NCHW_SHAPE, NCHW_STRIDE)
    arg261_1 = _require_f32_tensor("arg261_1", arg261_1, GATE_SHAPE, GATE_STRIDE)

    devices = {x.device for x in (getitem_129, arg262_1, add_62, arg258_1, arg261_1)}
    if len(devices) != 1:
        raise ValueError("all tensor inputs must be on the same device")
    return getitem_129, arg262_1, add_62, arg258_1, arg261_1


def oracle_forward(inputs):
    """Run the full Repro.forward computation with the same output shape/dtype/stride."""
    (
        getitem_129,
        arg262_1,
        add_62,
        arg258_1,
        arg261_1,
    ) = _validate_inputs(inputs)

    if getitem_129.device.type != "cuda":
        return _torch_oracle(inputs)
    if triton is None or libdevice is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    partial = torch.empty_strided((N, C), (C, 1), device=getitem_129.device, dtype=torch.float32)
    _spatial_gate_partial_kernel[(triton.cdiv(C, REDUCE_BLOCK_C), N)](
        getitem_129,
        arg262_1,
        add_62,
        arg258_1,
        arg261_1,
        partial,
        BLOCK_HW=SPATIAL_BLOCK,
        BLOCK_C=REDUCE_BLOCK_C,
        num_warps=8,
        num_stages=1,
    )

    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=getitem_129.device, dtype=torch.float32)
    _final_channel_sum_kernel[(triton.cdiv(C, REDUCE_BLOCK_C),)](
        partial,
        out,
        BLOCK_N=FINAL_BLOCK_N,
        BLOCK_C=REDUCE_BLOCK_C,
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
