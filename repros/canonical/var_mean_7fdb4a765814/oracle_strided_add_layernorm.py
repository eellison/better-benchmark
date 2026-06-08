"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete strided residual add, population var_mean over hidden size 768, eps=1e-6 rsqrt normalization, fp32 affine scale/bias, contiguous clone materialization, and final [32768, 768] view in one row-reduction Triton kernel, whereas Inductor currently lowers the decomposed add/var_mean/affine/clone/view graph through its generic normalization reduction schedule for the transposed residual input; Inductor cannot do this today because its scheduler/codegen lacks a fixed-hidden layernorm template that keeps the non-contiguous residual add tile live through the var_mean epilogue and writes the final contiguous view directly; the fix is SCHEDULER_FUSION: add a benchmark-gated layernorm row schedule for contiguous-plus-transposed residual inputs that fuses the reduction, affine epilogue, and output materialization."""
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

ROWS = 32768
BATCH = 128
SEQ = 256
HIDDEN = 768
EPS = 1.0e-6
BLOCK_H = 1024
BLOCK_M = 4

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
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm, residual, weight, bias, view_shape = inputs
    addmm_t = _require_f32_tensor(
        "addmm_47",
        addmm,
        (ROWS, HIDDEN),
        (HIDDEN, 1),
    )
    residual_t = _require_f32_tensor(
        "add_80",
        residual,
        (BATCH, SEQ, HIDDEN),
        (SEQ * HIDDEN, 1, SEQ),
    )
    weight_t = _require_f32_tensor("arg148_1", weight, (HIDDEN,), (1,))
    bias_t = _require_f32_tensor("arg149_1", bias, (HIDDEN,), (1,))

    if _shape_tuple(view_shape) != (BATCH, SEQ, HIDDEN):
        raise ValueError(f"unexpected view shape parameter: {view_shape!r}")

    device = addmm_t.device
    if any(t.device != device for t in (residual_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm_t, residual_t, weight_t, bias_t


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _strided_add_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        rows_total: tl.constexpr,
        seq: tl.constexpr,
        eps: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_h)
        row_mask = rows < rows_total
        col_mask = cols < hidden
        mask = row_mask[:, None] & col_mask[None, :]

        batch = rows // seq
        token = rows - batch * seq
        addmm_offsets = rows[:, None] * hidden + cols[None, :]
        residual_offsets = batch[:, None] * (seq * hidden) + token[:, None] + cols[None, :] * seq

        x = (
            tl.load(addmm_ptr + addmm_offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
        )
        x_masked = tl.where(mask, x, 0.0)
        mean = tl.sum(x_masked, axis=1) / hidden
        centered = x - mean[:, None]
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden
        inv_std = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        out = centered * inv_std[:, None] * weight[None, :] + bias[None, :]
        tl.store(out_ptr + addmm_offsets, out, mask=mask)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_strided_add_layernorm.py")

    addmm, residual, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    _strided_add_layernorm_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        addmm,
        residual,
        weight,
        bias,
        output,
        hidden=HIDDEN,
        rows_total=ROWS,
        seq=SEQ,
        eps=EPS,
        block_m=BLOCK_M,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=3,
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
