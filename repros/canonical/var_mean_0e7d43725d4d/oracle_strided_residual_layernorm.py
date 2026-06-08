"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete strided-residual f32 LayerNorm scope in one shape-specialized Triton row kernel, including the `[32768,768] -> [128,256,768]` view, the non-contiguous residual add, population `var_mean(..., dim=2, correction=0, keepdim=True)` via Welford state, `libdevice.rsqrt(var + 1e-6)`, affine scale/bias, final contiguous `[32768,768]` view, and sibling `rsqrt * (1/768)` side output, whereas Inductor emits a full-scope but generic fused Welford reduction for the same graph with extra indexing and reduction-template overhead; Inductor cannot do this today because the normalization scheduler/codegen path does not select a guarded fixed-hidden strided-residual LayerNorm schedule that keeps the producer tile through the affine and side-output epilogues; the fix is SCHEDULER_FUSION: add a benchmark-gated LayerNorm template for this producer stride pattern and side-output contract while preserving the exact Welford and `libdevice.rsqrt` lowering."""
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
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    triton_helpers = None
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


ROWS = 32768
BATCH = 128
SEQ_LEN = 256
HIDDEN = 768
ADDMM_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
ADDMM_STRIDE = (HIDDEN, 1)
RESIDUAL_STRIDE = (SEQ_LEN * HIDDEN, 1, SEQ_LEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, SEQ_LEN, 1)
SIDE_STRIDE = (SEQ_LEN, 1, 1)
EPS = 1.0e-6
SIDE_SCALE = 0.0013020833333333333
BLOCK_H = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _strided_residual_layernorm_kernel(
        residual_ptr,
        addmm_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        side_ptr,
        total_rows: tl.constexpr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        side_scale: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, BLOCK_H)[None, :]
        row_mask_1d = row_ids < total_rows
        row_mask = row_mask_1d[:, None]
        col_mask = cols < hidden
        mask = row_mask & col_mask

        seq = row_ids % seq_len
        batch = row_ids // seq_len
        addmm_offsets = rows * hidden + cols
        residual_offsets = batch[:, None] * (seq_len * hidden) + seq[:, None] + cols * seq_len

        residual = tl.load(
            residual_ptr + residual_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        addmm = tl.load(
            addmm_ptr + addmm_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        x = residual + addmm

        values = tl.where(mask, x, 0.0)
        m2_vec = tl.zeros([ROW_BLOCK, BLOCK_H], tl.float32)
        weight_vec = tl.where(mask, 1.0, 0.0)
        mean_1d, m2_1d, _ = triton_helpers.welford(values, m2_vec, weight_vec, 1)
        mean = mean_1d[:, None]
        variance_1d = m2_1d / hidden
        invstd_1d = libdevice.rsqrt(variance_1d + eps)
        invstd = invstd_1d[:, None]

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
        output = ((x - mean) * invstd) * weight + bias
        tl.store(output_ptr + addmm_offsets, output, mask=mask)
        tl.store(side_ptr + row_ids, invstd_1d * side_scale, mask=row_mask_1d)


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
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_47", inputs[0], ADDMM_SHAPE, torch.float32, ADDMM_STRIDE)
    residual = _require_tensor(
        "add_80",
        inputs[1],
        RESIDUAL_SHAPE,
        torch.float32,
        RESIDUAL_STRIDE,
    )
    weight = _require_tensor("arg148_1", inputs[2], (HIDDEN,), torch.float32, (1,))
    bias = _require_tensor("arg149_1", inputs[3], (HIDDEN,), torch.float32, (1,))

    if _shape_tuple(inputs[4]) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected view shape parameter: {inputs[4]!r}")

    device = addmm.device
    if any(value.device != device for value in (residual, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm, residual, weight, bias


def oracle_forward(inputs):
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    This preserves the captured Inductor math: residual + viewed addmm in f32,
    Welford population variance (`correction=0`), `libdevice.rsqrt(var + 1e-6)`,
    f32 affine epilogue, and side output as `rsqrt * 1/768`.
    """
    if triton is None or triton_helpers is None or libdevice is None:
        raise RuntimeError("Triton is required for oracle_strided_residual_layernorm.py")

    addmm, residual, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _strided_residual_layernorm_kernel[grid](
        residual,
        addmm,
        weight,
        bias,
        output,
        side,
        total_rows=ROWS,
        seq_len=SEQ_LEN,
        hidden=HIDDEN,
        eps=EPS,
        side_scale=SIDE_SCALE,
        BLOCK_H=BLOCK_H,
    )
    return output, side


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
