"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete masked f32 product, returns it as the required transposed view, and reduces the same product with the same 128-row then 32-partial ordering as generated Inductor, whereas Inductor already emits this two-stage fused product/reduction form; Inductor cannot close much more here because the full-scope result must stream both inputs and write the full product tensor, so the fix is BANDWIDTH_BOUND: keep this pattern on the existing fused reduction template and treat remaining variance as memory traffic plus launch overhead."""
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
    has_stochastic_ops,
)


M = 4096
N = 2560
ROW_BLOCK = 128
PARTIAL_ROWS = 32
FIRST_X_BLOCK = 128
FIRST_R_BLOCK = 16
FINAL_X_BLOCK = 256
FINAL_R_BLOCK = 32
_WORKSPACES = {}


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _product_partial_kernel(
        values_ptr,
        mask_ptr,
        product_ptr,
        partial_ptr,
        MATRIX_N: tl.constexpr,
        BATCH_STRIDE: tl.constexpr,
        ROWS_IN_CHUNK: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        cols = xindex % MATRIX_N
        batch = xindex // MATRIX_N
        rbase = tl.arange(0, RBLOCK)[None, :]

        acc = tl.full([XBLOCK, RBLOCK], 0.0, tl.float32)
        for roffset in tl.range(0, ROWS_IN_CHUNK, RBLOCK):
            ridx = roffset + rbase
            rmask = ridx < ROWS_IN_CHUNK
            offsets = cols + MATRIX_N * ridx + BATCH_STRIDE * batch

            values = tl.load(values_ptr + offsets, rmask, eviction_policy="evict_first", other=0.0)
            mask_i1 = tl.load(mask_ptr + offsets, rmask, eviction_policy="evict_first", other=0).to(tl.int1)
            mask_f32 = mask_i1.to(tl.float32)
            scale = tl.full([1, 1], 1.1111111111111112, tl.float32)
            scaled_mask = mask_f32 * scale
            product = values * scaled_mask

            acc_next = acc + product
            acc = tl.where(rmask, acc_next, acc)
            tl.store(product_ptr + offsets, product, rmask)

        partial = tl.sum(acc, axis=1)[:, None]
        tl.store(partial_ptr + xindex, partial, None)

    @triton.jit
    def _final_sum_kernel(
        partial_ptr,
        out_ptr,
        MATRIX_N: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < MATRIX_N
        ridx = tl.arange(0, RBLOCK)[None, :]

        partials = tl.load(
            partial_ptr + xindex + MATRIX_N * ridx,
            xmask,
            eviction_policy="evict_first",
            other=0.0,
        )
        partials = tl.where(xmask, partials, 0.0)
        result = tl.sum(partials, axis=1)[:, None]
        tl.store(out_ptr + xindex, result, xmask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 4:
        raise ValueError(f"expected 4 inputs, got {len(inputs)}")

    mask, values, shape0, shape1 = inputs
    if not isinstance(mask, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise TypeError("expected tensor inputs in positions 0 and 1")
    if mask.device.type != "cuda" or values.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensors")
    if mask.dtype != torch.bool:
        raise TypeError(f"expected bool mask, got {mask.dtype}")
    if values.dtype != torch.float32:
        raise TypeError(f"expected f32 values, got {values.dtype}")
    if tuple(mask.shape) != (32, 128, 2560) or tuple(values.shape) != (32, 128, 2560):
        raise ValueError(f"unexpected tensor shapes: {tuple(mask.shape)} {tuple(values.shape)}")
    if mask.stride() != (327680, 2560, 1) or values.stride() != (327680, 2560, 1):
        raise ValueError("oracle expects the captured contiguous input strides")
    if tuple(shape0) != (M, N) or tuple(shape1) != (N,):
        raise ValueError(f"unexpected shape params: {shape0} {shape1}")
    return mask, values


def _get_workspace(device):
    key = (device.type, device.index)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        product_base = torch.empty_strided((M, N), (N, 1), device=device, dtype=torch.float32)
        partials = torch.empty_strided((PARTIAL_ROWS, N), (N, 1), device=device, dtype=torch.float32)
        summed = torch.empty_strided((N,), (1,), device=device, dtype=torch.float32)
        product_view = product_base.as_strided((N, M), (1, N))
        workspace = (product_base, product_view, partials, summed)
        _WORKSPACES[key] = workspace
    return workspace


def oracle_forward(inputs):
    """Run the full Repro.forward computation with the same output structure."""
    mask, values = _validate_inputs(inputs)

    product_base, product_view, partials, summed = _get_workspace(values.device)

    _product_partial_kernel[(triton.cdiv(PARTIAL_ROWS * N, FIRST_X_BLOCK),)](
        values,
        mask,
        product_base,
        partials,
        MATRIX_N=N,
        BATCH_STRIDE=ROW_BLOCK * N,
        ROWS_IN_CHUNK=ROW_BLOCK,
        XBLOCK=FIRST_X_BLOCK,
        RBLOCK=FIRST_R_BLOCK,
        num_warps=2,
        num_stages=1,
        sanitize_overflow=False,
    )
    _final_sum_kernel[(triton.cdiv(N, FINAL_X_BLOCK),)](
        partials,
        summed,
        MATRIX_N=N,
        XBLOCK=FINAL_X_BLOCK,
        RBLOCK=FINAL_R_BLOCK,
        num_warps=8,
        num_stages=1,
        sanitize_overflow=False,
    )

    return product_view, summed


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
