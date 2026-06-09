"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DINOv2 residual LayerScale plus affine LayerNorm return value by sinking `select(..., 1, 0).clone()` through the row-independent reshape, scaled MLP residual add, fp32 population var_mean, and affine epilogue, so only the 128 live class-token rows are reduced and stored, whereas Inductor currently materializes Welford mean/M2 statistics for all `128*1370` token rows before a second kernel writes the selected class-token clone; Inductor cannot do this today because its normalization scheduler does not commute a constant token select backward through row-local LayerNorm and its producer expression to narrow the row domain; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to push constant token selects through row-local normalization templates and eliminate dead token rows before scheduling the Welford reduction."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
BATCH = 128
SEQ_LEN = 1370
HIDDEN = 768
ROWS = BATCH * SEQ_LEN
EPS = 1.0e-6
BLOCK_H = 1024
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"

if triton is not None:
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice

    @triton.jit
    def _selected_class_token_layernorm_kernel(
        addmm_ptr,
        layerscale_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        batch = tl.program_id(0)
        cols = tl.arange(0, block_h)
        col_mask = cols < hidden
        row = batch * seq_len
        offsets = row * hidden + cols

        residual = tl.load(residual_ptr + offsets, mask=col_mask, other=0.0)
        addmm = tl.load(addmm_ptr + offsets, mask=col_mask, other=0.0)
        layerscale = tl.load(layerscale_ptr + cols, mask=col_mask, other=0.0)
        values = residual + addmm * layerscale

        mean_acc = tl.zeros([block_h], tl.float32)
        m2_acc = tl.zeros([block_h], tl.float32)
        weight_acc = tl.zeros([block_h], tl.float32)
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            values, mean_acc, m2_acc, weight_acc, True
        )
        mean_acc = tl.where(col_mask, mean_next, mean_acc)
        m2_acc = tl.where(col_mask, m2_next, m2_acc)
        weight_acc = tl.where(col_mask, weight_next, weight_acc)
        mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 0)

        centered = values - mean
        variance = m2 / 768.0
        invstd = libdevice.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0)
        output = centered * invstd * weight + bias
        tl.store(output_ptr + batch * hidden + cols, output, mask=col_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm_47 = _require_tensor(
        "addmm_47",
        inputs[0],
        (ROWS, HIDDEN),
        torch.float32,
        (HIDDEN, 1),
    )
    layerscale = _require_tensor("arg172_1", inputs[1], (HIDDEN,), torch.float32, (1,))
    residual = _require_tensor(
        "add_80",
        inputs[2],
        (BATCH, SEQ_LEN, HIDDEN),
        torch.float32,
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
    )
    weight = _require_tensor("arg173_1", inputs[3], (HIDDEN,), torch.float32, (1,))
    bias = _require_tensor("arg174_1", inputs[4], (HIDDEN,), torch.float32, (1,))

    shape_param = _shape_tuple(inputs[5])
    if shape_param != (BATCH, SEQ_LEN, HIDDEN):
        raise ValueError(f"_shape_param_0 is {shape_param}, expected {(BATCH, SEQ_LEN, HIDDEN)}")

    device = addmm_47.device
    for tensor in (layerscale, residual, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm_47, layerscale, residual, weight, bias


@oracle_impl(hardware="H100", shapes="(T([175360, 768], f32), T([768], f32), T([128, 1370, 768], f32), T([768], f32), T([768], f32), S([128, 1370, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
    """Run the complete Repro.forward computation for the returned class token.

    SCOPE INVARIANT: accepts the same six inputs as Repro.forward() and returns
    the same single contiguous float32 `[128, 768]` output. The reshape,
    LayerScale multiply, residual add, row-wise Welford var_mean, affine
    LayerNorm, select, and clone are represented by direct indexing into token
    zero for each batch.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_selected_class_token_layernorm.py")

    addmm_47, layerscale, residual, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=addmm_47.device,
        dtype=torch.float32,
    )
    _selected_class_token_layernorm_kernel[(BATCH,)](
        addmm_47,
        layerscale,
        residual,
        weight,
        bias,
        output,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=8,
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
