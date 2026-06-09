"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Visformer attention softmax-backward scale tail returned by Repro.forward, including the metadata-only `[768,196,196] -> [128,6,196,196]` reshape, f32 product, last-dimension row sum, explicit `fma.rn.f32` softmax-gradient epilogue, final f32 multiply by 0.125, and contiguous `[768,196,196]` output view in one Triton row kernel, whereas Inductor is expected to lower this compact K=196 row reduction and epilogue to the same CUDAGraph-measured memory-traffic floor; Inductor cannot materially improve this through a local scheduler-fusion change if the full-scope row oracle is at floor because the required two f32 reads, one row reduction, fma epilogue, scale, and one f32 write dominate; the fix is BANDWIDTH_BOUND: record as at_floor unless the benchmark shows this row-reduction epilogue oracle materially beats torch.compile, in which case the actionable class is SCHEDULER_FUSION for preserving this reduction epilogue as one schedule."""
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
HEADS = 6
Q_LEN = 196
K_LEN = 196
FLAT_BH = BATCH * HEADS
N_ROWS = FLAT_BH * Q_LEN

VIEW_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
OUT_SHAPE = (FLAT_BH, Q_LEN, K_LEN)
VIEW_STRIDE = (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1)
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)
BLOCK_K = 256
SCALE = 0.125

if triton is not None:

    @triton.jit
    def _fma_rn_f32(a, b, c):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            constraints="=f,f,f,f",
            args=[a, b, c],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _visformer_softmax_backward_scale_kernel(
        bmm_ptr,
        div_ptr,
        out_ptr,
        N_ROWS_: tl.constexpr,
        K_LEN_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        cols = tl.arange(0, BLOCK_N)[None, :]
        row_mask = rows < N_ROWS_
        col_mask = cols < K_LEN_
        mask = row_mask & col_mask
        offsets = rows * K_LEN_ + cols

        bmm = tl.load(
            bmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        div = tl.load(
            div_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        product = bmm * div
        row_sum = tl.sum(tl.where(col_mask, product, 0.0), axis=1)[:, None].to(tl.float32)
        fma = _fma_rn_f32(-div, row_sum, product)
        out = fma * SCALE_
        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected a shape sequence, got {value!r}") from exc


def _require_tensor(
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


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    bmm_45 = _require_tensor("bmm_45", inputs[0], OUT_SHAPE, OUT_STRIDE)
    div = _require_tensor("div", inputs[1], VIEW_SHAPE, VIEW_STRIDE)

    shape0 = _shape_tuple(inputs[2])
    shape1 = _shape_tuple(inputs[3])
    if shape0 != VIEW_SHAPE:
        raise ValueError(f"_shape_param_0 is {shape0}, expected {VIEW_SHAPE}")
    if shape1 != OUT_SHAPE:
        raise ValueError(f"_shape_param_1 is {shape1}, expected {OUT_SHAPE}")
    if bmm_45.device != div.device:
        raise ValueError("bmm_45 and div must be on the same CUDA device")
    return bmm_45, div


@oracle_impl(hardware="H100", shapes="(T([768, 196, 196], f32), T([128, 6, 196, 196], f32), S([128, 6, 196, 196]), S([768, 196, 196]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same four inputs as Repro.forward() and returns
    the complete f32 [768,196,196] output with the eager contiguous stride.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_visformer_softmax_backward_scale.py")

    bmm_45, div = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_45.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(N_ROWS, meta["BLOCK_M"]),)
    _visformer_softmax_backward_scale_kernel[grid](
        bmm_45,
        div,
        out,
        N_ROWS_=N_ROWS,
        K_LEN_=K_LEN,
        SCALE_=SCALE,
        BLOCK_N=BLOCK_K,
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
