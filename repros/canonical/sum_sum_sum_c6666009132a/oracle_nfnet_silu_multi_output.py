"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete `sum_sum_sum_c6666009132a` NFNet backward fragment by streaming the shared scaled-input, gate sigmoid, explicit natural-exp SiLU-derivative producer once into per-`(N,C)` spatial summaries for both returned channel reductions, whereas Inductor materializes `add_tensor_3`, materializes the spatial `[128,512,1,1]` sum, then launches separate channel reductions for the sigmoid-gradient output and sibling `add_tensor_3` sum; Inductor cannot do this today because its algebraic simplifier/reduction codegen does not flatten this linear `sum([2,3]) -> sigmoid-derivative multiply -> sum([0,2,3])` chain into the same multi-output reduction as the sibling channel sum while preserving the captured exp/reciprocal formulation; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to reassociate such dependent reductions and emit one multi-accumulator channel-reduction schedule over the shared fused producer. Numeric lowering matches generated Inductor: captured `aten.sigmoid` uses `tl.sigmoid`, and the explicit captured `aten.exp` path uses `libdevice.exp` before reciprocal."""
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
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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
    has_stochastic_ops,
)


N = 128
C = 512
H = 28
W = 28
HW = H * W

SCALE_GETITEM = 0.9805806756909201
POINTWISE_SCALE = 0.2

SPATIAL_BLOCK_C = 8
SPATIAL_BLOCK_HW = 256
FINAL_BLOCK_C = 16
FINAL_BLOCK_N = 128


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _spatial_summaries_kernel(
        getitem_ptr,
        gate_arg_ptr,
        arg196_ptr,
        arg181_ptr,
        mul645_ptr,
        gate_nc_ptr,
        add3_nc_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        SCALE_GETITEM_: tl.constexpr,
        POINTWISE_SCALE_: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n = tl.program_id(0)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c < C_

        gate_arg = tl.load(gate_arg_ptr + n * C_ + c, mask=c_mask, other=0.0).to(tl.float32)
        sigmoid_default = tl.sigmoid(gate_arg)

        acc_spatial_grad = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_add3 = tl.zeros((BLOCK_C,), dtype=tl.float32)

        hw_offsets_base = tl.arange(0, BLOCK_HW)
        for hw_base in tl.static_range(0, HW_, BLOCK_HW):
            hw = hw_base + hw_offsets_base
            hw_mask = hw < HW_
            mask = c_mask[:, None] & hw_mask[None, :]
            offsets = n * C_ * HW_ + c[:, None] * HW_ + hw[None, :]

            getitem_186 = tl.load(getitem_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            arg196_1 = tl.load(arg196_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            arg181_1 = tl.load(arg181_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            mul_645 = tl.load(mul645_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

            mul_tensor = getitem_186 * SCALE_GETITEM_
            mul_tensor_1 = arg196_1 * sigmoid_default[:, None]
            mul_tensor_2 = mul_tensor_1 * 2.0
            mul_tensor_3 = mul_tensor_2 * POINTWISE_SCALE_
            add_tensor = mul_tensor_3 + arg181_1

            neg_default = -add_tensor
            exp_default = libdevice.exp(neg_default)
            add_tensor_1 = exp_default + 1.0
            reciprocal_default = 1.0 / add_tensor_1
            mul_tensor_4 = reciprocal_default * 1.0
            mul_tensor_5 = mul_tensor * mul_tensor_4
            sub_tensor = 1.0 - mul_tensor_4
            mul_tensor_6 = add_tensor * sub_tensor
            add_tensor_2 = mul_tensor_6 + 1.0
            mul_tensor_7 = mul_tensor_5 * add_tensor_2
            add_tensor_3 = mul_645 + mul_tensor_7

            mul_tensor_8 = add_tensor_3 * POINTWISE_SCALE_
            mul_tensor_9 = mul_tensor_8 * 2.0
            mul_tensor_10 = mul_tensor_9 * arg196_1

            acc_spatial_grad += tl.sum(tl.where(mask, mul_tensor_10, 0.0), axis=1)
            acc_add3 += tl.sum(tl.where(mask, add_tensor_3, 0.0), axis=1)

        sub_tensor_1 = 1.0 - sigmoid_default
        mul_tensor_11 = sigmoid_default * sub_tensor_1
        mul_tensor_12 = acc_spatial_grad * mul_tensor_11

        out_offsets = n * C_ + c
        tl.store(gate_nc_ptr + out_offsets, mul_tensor_12, mask=c_mask)
        tl.store(add3_nc_ptr + out_offsets, acc_add3, mask=c_mask)

    @triton.jit
    def _finalize_channels_kernel(
        gate_nc_ptr,
        add3_nc_ptr,
        out_gate_ptr,
        out_add3_ptr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        block = tl.program_id(0)
        n = tl.arange(0, BLOCK_N)
        c = block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (n[:, None] < N_) & (c[None, :] < C_)
        offsets = n[:, None] * C_ + c[None, :]

        gate_vals = tl.load(gate_nc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add3_vals = tl.load(add3_nc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        gate_sum = tl.sum(gate_vals, axis=0)
        add3_sum = tl.sum(add3_vals, axis=0)

        c_mask = c < C_
        tl.store(out_gate_ptr + c, gate_sum, mask=c_mask)
        tl.store(out_add3_ptr + c, add3_sum, mask=c_mask)


def _expect_tensor(name: str, value: Any, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} expected shape={shape}, got {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} expected torch.float32, got {value.dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{REPRO_ID} expects CUDA inputs")
    if not value.is_contiguous():
        raise ValueError(f"{name} expected captured contiguous layout, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    getitem_186, arg199_1, arg196_1, arg181_1, mul_645 = inputs
    expected_4d = (N, C, H, W)
    expected_gate = (N, C, 1, 1)
    return (
        _expect_tensor("getitem_186", getitem_186, expected_4d),
        _expect_tensor("arg199_1", arg199_1, expected_gate),
        _expect_tensor("arg196_1", arg196_1, expected_4d),
        _expect_tensor("arg181_1", arg181_1, expected_4d),
        _expect_tensor("mul_645", mul_645, expected_4d),
    )


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward computation with fused multi-output reductions."""
    if triton is None or tl is None or libdevice is None:
        raise RuntimeError("Triton with libdevice is required for this oracle")

    getitem_186, arg199_1, arg196_1, arg181_1, mul_645 = _validate_inputs(inputs)
    device = getitem_186.device

    gate_nc = torch.empty((N, C), device=device, dtype=torch.float32)
    add3_nc = torch.empty((N, C), device=device, dtype=torch.float32)

    _spatial_summaries_kernel[(N, triton.cdiv(C, SPATIAL_BLOCK_C))](
        getitem_186,
        arg199_1,
        arg196_1,
        arg181_1,
        mul_645,
        gate_nc,
        add3_nc,
        C_=C,
        HW_=HW,
        SCALE_GETITEM_=SCALE_GETITEM,
        POINTWISE_SCALE_=POINTWISE_SCALE,
        BLOCK_C=SPATIAL_BLOCK_C,
        BLOCK_HW=SPATIAL_BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )

    out_gate = torch.empty((C,), device=device, dtype=torch.float32)
    out_add3 = torch.empty((C,), device=device, dtype=torch.float32)

    _finalize_channels_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        gate_nc,
        add3_nc,
        out_gate,
        out_add3,
        N_=N,
        C_=C,
        BLOCK_N=FINAL_BLOCK_N,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )

    return out_gate, out_add3


# --- CLI entry point ---
def main() -> None:
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
