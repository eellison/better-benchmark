"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BEiT inference gamma-residual-LayerNorm scope in one shape-specialized Triton row kernel, including the `[25216,768] -> [128,197,768]` metadata view, gamma multiplication before residual add, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 libdevice rsqrt, affine scale/bias, and final contiguous `[25216,768]` output, whereas tuned Inductor already emits the same norm-template row-reduction work for this fixed hidden size; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, recompute fusion, or a new narrow pattern because the remaining work is dominated by required reads of activation/residual/gamma/affine tensors, one row reduction, rsqrt, and output traffic; the fix is BANDWIDTH_BOUND: record this as a full-scope floor unless broader normalization-template, launch, or memory-traffic work moves both paths."""
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
except ImportError:
    triton = None
    tl = None
    libdevice = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 25216
BATCH = 128
TOKENS = 197
HIDDEN = 768
INPUT_SHAPE = (ROWS, HIDDEN)
VIEW_SHAPE = (BATCH, TOKENS, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
EPS = 1.0e-6
BLOCK_H = 1024

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package. The shared harness owns timing so
# CUDA graph capture, GPU locking, and interleaved measurement are preserved.
from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


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
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_45", inputs[0], INPUT_SHAPE, torch.float32)
    gamma = _require_tensor("arg202_1", inputs[1], (HIDDEN,), torch.float32)
    residual = _require_tensor("add_76", inputs[2], VIEW_SHAPE, torch.float32)
    weight = _require_tensor("arg214_1", inputs[3], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg215_1", inputs[4], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[5]) != VIEW_SHAPE:
        raise ValueError(f"unexpected first reshape shape parameter: {inputs[5]!r}")
    if _shape_tuple(inputs[6]) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected final reshape shape parameter: {inputs[6]!r}")

    device = addmm.device
    if any(t.device != device for t in (gamma, residual, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return addmm, gamma, residual, weight, bias


if triton is not None:

    @triton.jit
    def _beit_layernorm_kernel(
        addmm_ptr,
        gamma_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < hidden
        offsets = row * hidden + cols

        addmm = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        gamma = tl.load(
            gamma_ptr + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        value = residual + gamma * addmm

        reduced_value = tl.where(mask, value, 0.0)
        mean = tl.sum(reduced_value, axis=0) / hidden
        centered = value - mean
        reduced_centered = tl.where(mask, centered, 0.0)
        variance = tl.sum(reduced_centered * reduced_centered, axis=0) / hidden
        invstd = libdevice.rsqrt(variance + eps)

        weight = tl.load(
            weight_ptr + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        normalized = centered * invstd
        scaled = normalized * weight
        output = scaled + bias
        tl.store(out_ptr + offsets, output, mask=mask)


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, gamma, residual, weight, bias = _validate_inputs(inputs)
    reshape_default = torch.ops.aten.reshape.default(addmm, _shape_tuple(inputs[5]))
    mul_tensor = torch.ops.aten.mul.Tensor(gamma, reshape_default)
    add_tensor = torch.ops.aten.add.Tensor(residual, mul_tensor)
    var_mean = torch.ops.aten.var_mean.correction(
        add_tensor, [2], correction=0, keepdim=True
    )
    variance = var_mean[0]
    mean = var_mean[1]
    centered = torch.ops.aten.sub.Tensor(add_tensor, mean)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(centered, invstd)
    scaled = torch.ops.aten.mul.Tensor(normalized, weight)
    affine = torch.ops.aten.add.Tensor(scaled, bias)
    return torch.ops.aten.reshape.default(affine, _shape_tuple(inputs[6]))


@oracle_impl(hardware="H100", shapes="(T([25216, 768], f32), T([768], f32), T([128, 197, 768], f32), T([768], f32), T([768], f32), S([128, 197, 768]), S([25216, 768]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward gamma-residual LayerNorm scope."""
    addmm, gamma, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or libdevice is None:
        return _torch_full_scope(inputs)

    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    _beit_layernorm_kernel[(ROWS,)](
        addmm,
        gamma,
        residual,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK=BLOCK_H,
        num_warps=8,
        num_stages=3,
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
