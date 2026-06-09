"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full ALBERT fragment with one cooperative split-K Triton kernel that writes the scaled/reshaped bmm_47 backing tensor and emits all twelve f32[4096] row-chunk partial reductions, followed by one finalize kernel that combines the 32 partial chunks in the captured left-to-right add order, whereas Inductor currently launches the layout-producing clone/mul/permute kernel, twelve separate first-stage reduction kernels, and a final persistent reduction over their partial buffers; Inductor cannot do this today because its scheduler does not form one multi-source cooperative split-K reduction that also writes a view-equivalent side output and carries that side output's partial sum; the fix is COOPERATIVE_SPLIT_K: add codegen for fused multi-input split-K column reductions with structured side-output stores and shared finalization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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


ROWS = 4096
COLS = 4096
ROW_CHUNKS = 32
ROWS_PER_CHUNK = ROWS // ROW_CHUNKS
BMM_SHAPE = (512, 512, 64)
VIEW_4D_SHAPE = (8, 64, 512, 64)
VIEW_3D_SHAPE = (8, 512, 4096)
FLAT_SHAPE = (ROWS, COLS)
SUM_SHAPE = (COLS,)
TRANSPOSE_STRIDE = (1, COLS)
SCALE = 0.3535533905932738


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
            triton.Config({"BLOCK_C": 8}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 16}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 32}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _partial_layout_multi_sum_kernel(
        in0_ptr,
        in1_ptr,
        in2_ptr,
        in3_ptr,
        in4_ptr,
        in5_ptr,
        in6_ptr,
        in7_ptr,
        in8_ptr,
        in9_ptr,
        in10_ptr,
        bmm_ptr,
        out_base_ptr,
        partials_ptr,
        COLS_: tl.constexpr,
        SCALE_: tl.constexpr,
        ROWS_PER_CHUNK_: tl.constexpr,
        ROW_CHUNKS_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        col_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        chunk = tl.program_id(1)
        col_mask = col_offsets < COLS_
        rows = chunk * ROWS_PER_CHUNK_ + tl.arange(0, ROWS_PER_CHUNK_)
        flat_offsets = rows[:, None] * COLS_ + col_offsets[None, :]
        mask = col_mask[None, :]
        partial_base = chunk * COLS_ + col_offsets
        scale = tl.full((), SCALE_, tl.float32)

        vals = tl.load(in0_ptr + flat_offsets, mask=mask, other=0.0)
        acc0 = tl.sum(vals, axis=0)
        vals = tl.load(in1_ptr + flat_offsets, mask=mask, other=0.0)
        acc1 = tl.sum(vals, axis=0)
        vals = tl.load(in2_ptr + flat_offsets, mask=mask, other=0.0)
        acc2 = tl.sum(vals, axis=0)
        vals = tl.load(in3_ptr + flat_offsets, mask=mask, other=0.0)
        acc3 = tl.sum(vals, axis=0)
        vals = tl.load(in4_ptr + flat_offsets, mask=mask, other=0.0)
        acc4 = tl.sum(vals, axis=0)
        vals = tl.load(in5_ptr + flat_offsets, mask=mask, other=0.0)
        acc5 = tl.sum(vals, axis=0)
        vals = tl.load(in6_ptr + flat_offsets, mask=mask, other=0.0)
        acc6 = tl.sum(vals, axis=0)
        vals = tl.load(in7_ptr + flat_offsets, mask=mask, other=0.0)
        acc7 = tl.sum(vals, axis=0)
        vals = tl.load(in8_ptr + flat_offsets, mask=mask, other=0.0)
        acc8 = tl.sum(vals, axis=0)
        vals = tl.load(in9_ptr + flat_offsets, mask=mask, other=0.0)
        acc9 = tl.sum(vals, axis=0)
        vals = tl.load(in10_ptr + flat_offsets, mask=mask, other=0.0)
        acc10 = tl.sum(vals, axis=0)

        # Captured layout: bmm_47.view(8,64,512,64) * scale,
        # permute(0,2,1,3).clone().view(4096,4096).
        bmm_batch = (rows[:, None] // 512) * 64 + (col_offsets[None, :] // 64)
        bmm_row = rows[:, None] - (rows[:, None] // 512) * 512
        bmm_col = col_offsets[None, :] - (col_offsets[None, :] // 64) * 64
        bmm_offsets = (bmm_batch * 512 + bmm_row) * 64 + bmm_col
        bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0)
        scaled = bmm_vals * scale
        tl.store(out_base_ptr + flat_offsets, scaled, mask=mask)
        acc11 = tl.sum(scaled, axis=0)

        tl.store(partials_ptr + partial_base, acc0, mask=col_mask)
        tl.store(partials_ptr + ROW_CHUNKS_ * COLS_ + partial_base, acc1, mask=col_mask)
        tl.store(partials_ptr + 2 * ROW_CHUNKS_ * COLS_ + partial_base, acc2, mask=col_mask)
        tl.store(partials_ptr + 3 * ROW_CHUNKS_ * COLS_ + partial_base, acc3, mask=col_mask)
        tl.store(partials_ptr + 4 * ROW_CHUNKS_ * COLS_ + partial_base, acc4, mask=col_mask)
        tl.store(partials_ptr + 5 * ROW_CHUNKS_ * COLS_ + partial_base, acc5, mask=col_mask)
        tl.store(partials_ptr + 6 * ROW_CHUNKS_ * COLS_ + partial_base, acc6, mask=col_mask)
        tl.store(partials_ptr + 7 * ROW_CHUNKS_ * COLS_ + partial_base, acc7, mask=col_mask)
        tl.store(partials_ptr + 8 * ROW_CHUNKS_ * COLS_ + partial_base, acc8, mask=col_mask)
        tl.store(partials_ptr + 9 * ROW_CHUNKS_ * COLS_ + partial_base, acc9, mask=col_mask)
        tl.store(partials_ptr + 10 * ROW_CHUNKS_ * COLS_ + partial_base, acc10, mask=col_mask)
        tl.store(partials_ptr + 11 * ROW_CHUNKS_ * COLS_ + partial_base, acc11, mask=col_mask)

    @triton.jit
    def _finalize_multi_sum_kernel(
        partials_ptr,
        out_sum_ptr,
        COLS_: tl.constexpr,
        ROW_CHUNKS_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        col_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        chunk_offsets = tl.arange(0, ROW_CHUNKS_)
        col_mask = col_offsets < COLS_
        mask = col_mask[None, :]
        offsets = chunk_offsets[:, None] * COLS_ + col_offsets[None, :]
        plane = ROW_CHUNKS_ * COLS_

        vals = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
        acc0 = tl.sum(vals, axis=0)
        vals = tl.load(partials_ptr + plane + offsets, mask=mask, other=0.0)
        acc1 = tl.sum(vals, axis=0)
        vals = tl.load(partials_ptr + 2 * plane + offsets, mask=mask, other=0.0)
        acc2 = tl.sum(vals, axis=0)
        vals = tl.load(partials_ptr + 3 * plane + offsets, mask=mask, other=0.0)
        acc3 = tl.sum(vals, axis=0)
        vals = tl.load(partials_ptr + 4 * plane + offsets, mask=mask, other=0.0)
        acc4 = tl.sum(vals, axis=0)
        vals = tl.load(partials_ptr + 5 * plane + offsets, mask=mask, other=0.0)
        acc5 = tl.sum(vals, axis=0)
        vals = tl.load(partials_ptr + 6 * plane + offsets, mask=mask, other=0.0)
        acc6 = tl.sum(vals, axis=0)
        vals = tl.load(partials_ptr + 7 * plane + offsets, mask=mask, other=0.0)
        acc7 = tl.sum(vals, axis=0)
        vals = tl.load(partials_ptr + 8 * plane + offsets, mask=mask, other=0.0)
        acc8 = tl.sum(vals, axis=0)
        vals = tl.load(partials_ptr + 9 * plane + offsets, mask=mask, other=0.0)
        acc9 = tl.sum(vals, axis=0)
        vals = tl.load(partials_ptr + 10 * plane + offsets, mask=mask, other=0.0)
        acc10 = tl.sum(vals, axis=0)
        vals = tl.load(partials_ptr + 11 * plane + offsets, mask=mask, other=0.0)
        acc11 = tl.sum(vals, axis=0)

        total = acc0 + acc1
        total = total + acc2
        total = total + acc3
        total = total + acc4
        total = total + acc5
        total = total + acc6
        total = total + acc7
        total = total + acc8
        total = total + acc9
        total = total + acc10
        total = total + acc11
        tl.store(out_sum_ptr + col_offsets, total, mask=col_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _expect_flat_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != FLAT_SHAPE:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match {FLAT_SHAPE}")
    if tuple(value.stride()) != (COLS, 1):
        raise ValueError(f"{name} stride {tuple(value.stride())} does not match {(COLS, 1)}")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have storage_offset=0, got {value.storage_offset()}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} dtype {value.dtype} does not match torch.float32")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    return value


def _expect_bmm_tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"bmm_47 must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != BMM_SHAPE:
        raise ValueError(f"bmm_47 shape {tuple(value.shape)} does not match {BMM_SHAPE}")
    if tuple(value.stride()) != (512 * 64, 64, 1):
        raise ValueError(f"bmm_47 stride {tuple(value.stride())} is not contiguous")
    if value.storage_offset() != 0:
        raise ValueError(f"bmm_47 must have storage_offset=0, got {value.storage_offset()}")
    if value.dtype != torch.float32:
        raise TypeError(f"bmm_47 dtype {value.dtype} does not match torch.float32")
    if value.device.type != "cuda":
        raise RuntimeError("bmm_47 must be a CUDA tensor")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 27:
        raise ValueError(f"{REPRO_ID} expects 27 inputs, got {len(inputs)}")

    flat_tensors = tuple(_expect_flat_tensor(f"input {i}", inputs[i]) for i in range(11))
    bmm = _expect_bmm_tensor(inputs[11])
    for i in range(12, 23):
        shape = _shape_tuple(inputs[i])
        if shape != SUM_SHAPE:
            raise ValueError(f"shape input {i} is {shape}, expected {SUM_SHAPE}")
    if _shape_tuple(inputs[23]) != VIEW_4D_SHAPE:
        raise ValueError(f"shape input 23 is {inputs[23]!r}, expected {VIEW_4D_SHAPE}")
    if _shape_tuple(inputs[24]) != VIEW_3D_SHAPE:
        raise ValueError(f"shape input 24 is {inputs[24]!r}, expected {VIEW_3D_SHAPE}")
    if _shape_tuple(inputs[25]) != FLAT_SHAPE:
        raise ValueError(f"shape input 25 is {inputs[25]!r}, expected {FLAT_SHAPE}")
    if _shape_tuple(inputs[26]) != SUM_SHAPE:
        raise ValueError(f"shape input 26 is {inputs[26]!r}, expected {SUM_SHAPE}")
    return (*flat_tensors, bmm)


@oracle_impl(hardware="H100", shapes="(T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([512, 512, 64], f32), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([8, 64, 512, 64]), S([8, 512, 4096]), S([4096, 4096]), S([4096]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same two outputs: a f32[4096,4096] transpose view with stride (1,4096) and
    the f32[4096] sum of the eleven input column reductions plus the scaled
    bmm-derived side-output column reduction.
    """
    tensors = _validate_inputs(inputs)
    (
        view_38,
        view_68,
        view_98,
        view_128,
        view_158,
        view_188,
        view_218,
        view_248,
        view_278,
        view_308,
        view_338,
        bmm_47,
    ) = tensors

    out_base = torch.empty_strided(
        FLAT_SHAPE,
        (COLS, 1),
        device=bmm_47.device,
        dtype=torch.float32,
    )
    out_sum = torch.empty(SUM_SHAPE, device=bmm_47.device, dtype=torch.float32)
    partials = torch.empty(
        (12, ROW_CHUNKS, COLS),
        device=bmm_47.device,
        dtype=torch.float32,
    )

    partial_grid = lambda meta: (triton.cdiv(COLS, meta["BLOCK_C"]), ROW_CHUNKS)
    _partial_layout_multi_sum_kernel[partial_grid](
        view_38,
        view_68,
        view_98,
        view_128,
        view_158,
        view_188,
        view_218,
        view_248,
        view_278,
        view_308,
        view_338,
        bmm_47,
        out_base,
        partials,
        COLS_=COLS,
        SCALE_=SCALE,
        ROWS_PER_CHUNK_=ROWS_PER_CHUNK,
        ROW_CHUNKS_=ROW_CHUNKS,
    )
    _finalize_multi_sum_kernel[(triton.cdiv(COLS, 16),)](
        partials,
        out_sum,
        COLS_=COLS,
        ROW_CHUNKS_=ROW_CHUNKS,
        BLOCK_C=16,
        num_warps=8,
    )
    return out_base.permute(1, 0), out_sum


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
