"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete LLaVA rotary-position scope for both returned tensors in one Triton layout/stencil kernel, deriving the permuted output layout directly, computing the shared position-frequency cos/sin values inside each head block, and loading the opposite half-dimension for rotate_half, whereas Inductor lowers the expand/clone/view/cos/sin producer and the two slice/cat rotary consumers as generic pointwise/layout work with separately scheduled shared intermediates; Inductor cannot do this today because its scheduler does not fuse a broadcast indexing/trig producer through layout views into multiple sibling stencil consumers while preserving the non-contiguous output stride; the fix is SCHEDULER_FUSION: add a RoPE layout-stencil fusion that sinks the position-frequency producer and half-rotation indexing into a single multi-output pointwise schedule."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)

SEQ = 512
HEADS = 32
HEAD_DIM = 128
HALF_DIM = 64
HIDDEN = HEADS * HEAD_DIM
OUT_SHAPE = (1, HEADS, SEQ, HEAD_DIM)
OUT_STRIDE = (SEQ * HIDDEN, HEAD_DIM, HIDDEN, 1)
CLASSIFICATION = "SCHEDULER_FUSION"
EXPECTED_SHAPE_PARAMS = (
    (1, 512, 4096),
    (1, 512, -1, 128),
    (1, 64, 1),
    (1, 1, 512),
    (1, 512, 2, 64),
    (1, 512, 128),
    (1, 512, 4096),
    (1, 512, -1, 128),
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _rope_pair_kernel(
        mm0_ptr,
        freq_ptr,
        mm1_ptr,
        out0_ptr,
        out1_ptr,
        BLOCK_H: tl.constexpr,
    ):
        pos = tl.program_id(0)
        head_block = tl.program_id(1)

        heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
        dims = tl.arange(0, 128)
        offsets = pos * 4096 + heads[:, None] * 128 + dims[None, :]
        mask = heads[:, None] < 32

        freq_idx = dims % 64
        phase = pos.to(tl.float32) * tl.load(freq_ptr + freq_idx).to(tl.float32)
        cos_vals = tl.cos(phase)
        sin_vals = tl.sin(phase)

        rot_offsets = tl.where(dims[None, :] < 64, offsets + 64, offsets - 64)
        rot_sign = tl.where(dims[None, :] < 64, -1.0, 1.0)

        x0 = tl.load(mm0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        r0 = tl.load(mm0_ptr + rot_offsets, mask=mask, other=0.0).to(tl.float32)
        y0 = x0 * cos_vals[None, :] + r0 * sin_vals[None, :] * rot_sign
        tl.store(out0_ptr + offsets, y0, mask=mask)

        x1 = tl.load(mm1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        r1 = tl.load(mm1_ptr + rot_offsets, mask=mask, other=0.0).to(tl.float32)
        y1 = x1 * cos_vals[None, :] + r1 * sin_vals[None, :] * rot_sign
        tl.store(out1_ptr + offsets, y1, mask=mask)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects 11 inputs, got {len(inputs)}")

    mm0, freq, mm1, *shape_params = inputs
    if not isinstance(mm0, torch.Tensor) or not isinstance(freq, torch.Tensor) or not isinstance(mm1, torch.Tensor):
        raise TypeError("first three inputs must be tensors")
    if tuple(mm0.shape) != (SEQ, HIDDEN) or tuple(mm1.shape) != (SEQ, HIDDEN):
        raise ValueError(f"unexpected input shapes: {tuple(mm0.shape)} and {tuple(mm1.shape)}")
    if tuple(freq.shape) != (HALF_DIM,):
        raise ValueError(f"unexpected frequency shape: {tuple(freq.shape)}")
    if mm0.dtype != torch.float16 or mm1.dtype != torch.float16 or freq.dtype != torch.float16:
        raise TypeError(f"expected fp16 tensors, got {mm0.dtype}, {freq.dtype}, {mm1.dtype}")
    if not mm0.is_cuda or not freq.is_cuda or not mm1.is_cuda:
        raise ValueError("oracle_rope_stencil.py expects CUDA tensor inputs")
    if mm0.device != freq.device or mm0.device != mm1.device:
        raise ValueError("all tensor inputs must be on the same device")
    if not mm0.is_contiguous() or not mm1.is_contiguous() or not freq.is_contiguous():
        raise ValueError("oracle_rope_stencil.py expects contiguous tensor inputs")

    for index, (actual, expected) in enumerate(zip(shape_params, EXPECTED_SHAPE_PARAMS), start=3):
        if tuple(actual) != expected:
            raise ValueError(f"unexpected shape parameter {index}: {actual!r}")
    return mm0, freq, mm1


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward RoPE scope with one Triton stencil kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_rope_stencil.py")

    mm0, freq, mm1 = _validate_inputs(inputs)
    out0 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=mm0.device, dtype=torch.float16)
    out1 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=mm0.device, dtype=torch.float16)

    _rope_pair_kernel[(SEQ, triton.cdiv(HEADS, 4))](
        mm0,
        freq,
        mm1,
        out0,
        out1,
        BLOCK_H=4,
        num_warps=4,
    )
    return (out0, out1)


def _check_layout(outputs: tuple[torch.Tensor, torch.Tensor]) -> bool:
    ok = True
    for index, output in enumerate(outputs):
        layout_ok = (
            tuple(output.shape) == OUT_SHAPE
            and output.stride() == OUT_STRIDE
            and output.dtype == torch.float16
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(output.shape)} stride={output.stride()} dtype={output.dtype})"
        )
        ok = ok and layout_ok
    return ok


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
        with torch.no_grad():
            layout_outputs = oracle_forward(inputs)
            torch.cuda.synchronize()
        ok = _check_layout(layout_outputs) and ok
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
