"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileNetV3 fp16 hard-sigmoid broadcast multiply as one shape-specialized contiguous Triton pointwise kernel with the explicit fp32 gate math, fp16 rounding boundary, and final fp16 multiply/store, whereas Inductor currently lowers the same cast/add/clamp/div/cast/mul scope through its generic flat pointwise codegen with comparable CUDAGraph replay time; Inductor cannot do this today because it has no guarded MobileNetV3 hard-sigmoid-broadcast-mul template that bypasses generic pointwise scheduler overhead while preserving the intermediate fp16 gate rounding; the fix is NEW_PATTERN: add a benchmark-gated pattern/template for singleton-spatial NCHW hard-sigmoid gates only if it consistently beats the generic pointwise schedule."""
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
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
        ],
        key=["N", "HW"],
    )
    @triton.jit
    def _flat_hardsigmoid_mul_kernel(
        gate_ptr,
        x_ptr,
        out_ptr,
        N: tl.constexpr,
        HW: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        gate_offsets = offsets // HW

        gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
        shifted = gate + 3.0
        clamped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
        scale = (clamped / 6.0).to(tl.float16)

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = x * scale
        tl.store(out_ptr + offsets, out, mask=mask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    gate, x = inputs
    if not isinstance(gate, torch.Tensor) or not isinstance(x, torch.Tensor):
        raise TypeError("both inputs must be tensors")
    if gate.device.type != "cuda" or x.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if gate.device != x.device:
        raise ValueError(f"inputs must be on the same device, got {gate.device} and {x.device}")
    if gate.dtype != torch.float16 or x.dtype != torch.float16:
        raise TypeError(f"expected fp16 inputs, got {gate.dtype} and {x.dtype}")
    if gate.dim() != 4 or x.dim() != 4:
        raise ValueError(f"expected rank-4 NCHW tensors, got {tuple(gate.shape)} and {tuple(x.shape)}")

    batch, channels, gate_h, gate_w = gate.shape
    x_batch, x_channels, height, width = x.shape
    if gate_h != 1 or gate_w != 1:
        raise ValueError(f"expected singleton spatial gate, got {tuple(gate.shape)}")
    if batch != x_batch or channels != x_channels:
        raise ValueError(f"gate and data shapes are incompatible: {tuple(gate.shape)} vs {tuple(x.shape)}")
    if not gate.is_contiguous() or not x.is_contiguous():
        raise ValueError("oracle expects the captured contiguous NCHW input layouts")

    return gate, x, int(height * width)


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
    gate, x, hw = _validate_inputs(inputs)
    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float16,
    )
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _flat_hardsigmoid_mul_kernel[grid](
        gate,
        x,
        output,
        N=n_elements,
        HW=hw,
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
                warmup=args.warmup, rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
