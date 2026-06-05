"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured GPT-J RoPE scope, including the `arg304_1` repeat/gather, split, coefficient duplication, both rotate-half branches over the first 64 head dimensions, tail preservation, concat assembly, and final permuted f32[1,16,128,256] output layouts in one Triton kernel, whereas Inductor lowers the same gather/expand/clone/rotate/cat/permute graph through generic pointwise and layout scheduling; Inductor is already within the harness floor for this small layout-heavy pointwise graph, so the oracle does not expose a meaningful remaining optimization gap; the fix is BANDWIDTH_BOUND: no Inductor change is justified for this repro beyond preserving the existing fused layout codegen path."""
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

BATCH = 1
SEQ = 128
HEADS = 16
HEAD_DIM = 256
ROTARY_DIM = 64
ROTARY_PAIRS = ROTARY_DIM // 2
TABLE_ROWS = 2048
OUT_SHAPE = (BATCH, HEADS, SEQ, HEAD_DIM)
OUT_STRIDE = (HEADS * SEQ * HEAD_DIM, HEAD_DIM, HEADS * HEAD_DIM, 1)
CLASSIFICATION = "BANDWIDTH_BOUND"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (  # noqa: E402
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_tensor(
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
    *,
    require_contiguous: bool = True,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype {value.dtype} does not match expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if require_contiguous and not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 14:
        raise ValueError(f"{REPRO_ID} expects 14 inputs, got {len(inputs)}")

    mm_108 = _validate_tensor(inputs[0], (SEQ, HEADS * HEAD_DIM), torch.float32, "mm_108")
    mm_109 = _validate_tensor(inputs[1], (SEQ, HEADS * HEAD_DIM), torch.float32, "mm_109")
    table = _validate_tensor(inputs[2], (TABLE_ROWS, ROTARY_DIM), torch.float32, "arg304_1")
    gather_idx = _validate_tensor(inputs[3], (BATCH, SEQ, ROTARY_DIM), torch.int64, "repeat_1")

    devices = {mm_108.device, mm_109.device, table.device, gather_idx.device}
    if len(devices) != 1:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    expected_shapes = (
        (BATCH, SEQ, HEADS * HEAD_DIM),
        (BATCH, SEQ, HEADS * HEAD_DIM),
        (BATCH, SEQ, HEADS, HEAD_DIM),
        (BATCH, SEQ, HEADS, HEAD_DIM),
        (BATCH, SEQ, 1, ROTARY_PAIRS, 2),
        (BATCH, SEQ, 1, ROTARY_DIM),
        (BATCH, SEQ, 1, ROTARY_PAIRS, 2),
        (BATCH, SEQ, 1, ROTARY_DIM),
        (BATCH, SEQ, HEADS, ROTARY_DIM),
        (BATCH, SEQ, HEADS, ROTARY_DIM),
    )
    for index, (value, expected) in enumerate(zip(inputs[4:], expected_shapes), start=4):
        if _shape_tuple(value) != expected:
            raise ValueError(f"shape parameter {index} value {_shape_tuple(value)} != {expected}")

    return mm_108, mm_109, table, gather_idx


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _gptj_gather_rope_kernel(
        mm108_ptr,
        mm109_ptr,
        table_ptr,
        index_ptr,
        out0_ptr,
        out1_ptr,
        N_ROWS: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        ROTARY_DIM_: tl.constexpr,
        ROTARY_PAIRS_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        dims = tl.arange(0, BLOCK_D)

        row_mask = rows < N_ROWS
        elem_mask = row_mask[:, None] & (dims[None, :] < HEAD_DIM_)

        pos = rows // NUM_HEADS
        head = rows - pos * NUM_HEADS
        base = pos[:, None] * (NUM_HEADS * HEAD_DIM_) + head[:, None] * HEAD_DIM_
        offsets = base + dims[None, :]

        rotary = dims < ROTARY_DIM_
        odd_dim = (dims & 1) == 1
        rot_dims = tl.where(odd_dim, dims - 1, dims + 1)
        rot_sign = tl.where(odd_dim, 1.0, -1.0)
        pair = dims // 2
        coeff_mask = row_mask[:, None] & rotary[None, :]

        idx0 = tl.load(
            index_ptr + pos[:, None] * ROTARY_DIM_ + pair[None, :],
            mask=coeff_mask,
            other=0,
        )
        idx1 = tl.load(
            index_ptr + pos[:, None] * ROTARY_DIM_ + ROTARY_PAIRS_ + pair[None, :],
            mask=coeff_mask,
            other=0,
        )
        coeff0 = tl.load(
            table_ptr + idx0 * ROTARY_DIM_ + pair[None, :],
            mask=coeff_mask,
            other=0.0,
        ).to(tl.float32)
        coeff1 = tl.load(
            table_ptr + idx1 * ROTARY_DIM_ + ROTARY_PAIRS_ + pair[None, :],
            mask=coeff_mask,
            other=0.0,
        ).to(tl.float32)

        x0 = tl.load(mm109_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        x0_rot = tl.load(
            mm109_ptr + base + rot_dims[None, :],
            mask=coeff_mask,
            other=0.0,
        ).to(tl.float32)
        y0 = tl.where(
            rotary[None, :],
            x0 * coeff1 + x0_rot * rot_sign[None, :] * coeff0,
            x0,
        )
        tl.store(out0_ptr + offsets, y0, mask=elem_mask)

        x1 = tl.load(mm108_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        x1_rot = tl.load(
            mm108_ptr + base + rot_dims[None, :],
            mask=coeff_mask,
            other=0.0,
        ).to(tl.float32)
        y1 = tl.where(
            rotary[None, :],
            x1 * coeff1 + x1_rot * rot_sign[None, :] * coeff0,
            x1,
        )
        tl.store(out1_ptr + offsets, y1, mask=elem_mask)


def oracle_forward(inputs):
    """Run the exact full Repro.forward scope with one fused Triton kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_gptj_rope_gather.py")

    mm_108, mm_109, table, gather_idx = _validate_inputs(inputs)
    out0 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=mm_109.device, dtype=torch.float32)
    out1 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=mm_108.device, dtype=torch.float32)

    n_rows = SEQ * HEADS
    block_rows = 1
    _gptj_gather_rope_kernel[(triton.cdiv(n_rows, block_rows),)](
        mm_108,
        mm_109,
        table,
        gather_idx,
        out0,
        out1,
        N_ROWS=n_rows,
        NUM_HEADS=HEADS,
        HEAD_DIM_=HEAD_DIM,
        ROTARY_DIM_=ROTARY_DIM,
        ROTARY_PAIRS_=ROTARY_PAIRS,
        BLOCK_ROWS=block_rows,
        BLOCK_D=HEAD_DIM,
        num_warps=8,
        num_stages=3,
    )
    return (out0, out1)


def _check_layout(outputs: tuple[torch.Tensor, ...]) -> bool:
    ok = True
    for i, output in enumerate(outputs):
        layout_ok = (
            tuple(output.shape) == OUT_SHAPE
            and tuple(output.stride()) == OUT_STRIDE
            and output.dtype == torch.float32
        )
        print(
            f"  output {i} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(output.shape)} dtype={output.dtype} stride={output.stride()})"
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
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        ok = ok and _check_layout(layout_out)
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
