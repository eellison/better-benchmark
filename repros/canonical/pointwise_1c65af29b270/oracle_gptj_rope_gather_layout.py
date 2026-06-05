"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle fuses the full GPT-J indexed RoPE scope, including the position-table gather, split coefficient reuse, duplicated-pair rotate-half arithmetic for both mm branches, tail concat preservation, and final permuted f32[1,16,128,256] output layouts into one Triton kernel, whereas Inductor's compiled graph is already within the harness floor for this small layout-heavy pointwise workload; the oracle does not expose a meaningful remaining scheduler or codegen gap because the required benchmark lands in the AT_FLOOR band; the fix is BANDWIDTH_BOUND: no Inductor change is justified for this repro beyond preserving the existing low-overhead fused layout path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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
ROWS = SEQ * HEADS
OUT_SHAPE = (BATCH, HEADS, SEQ, HEAD_DIM)
OUT_STRIDE = (HEADS * SEQ * HEAD_DIM, HEAD_DIM, HEADS * HEAD_DIM, 1)
SHAPE_PARAMS = (
    (1, 128, 4096),
    (1, 128, 16, 256),
    (1, 128, 1, 32, 2),
    (1, 128, 1, 64),
    (1, 128, 16, 64),
    (1, 128, 1, 32, 2),
    (1, 128, 1, 64),
    (1, 128, 4096),
    (1, 128, 16, 256),
    (1, 128, 1, 32, 2),
    (1, 128, 1, 64),
    (1, 128, 16, 64),
    (1, 128, 1, 32, 2),
    (1, 128, 1, 64),
)
CLASSIFICATION = "BANDWIDTH_BOUND"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (  # noqa: E402
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


def _validate_tensor(
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} {name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{REPRO_ID} {name} expects shape {shape}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{REPRO_ID} {name} expects dtype {dtype}, got {value.dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{REPRO_ID} {name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{REPRO_ID} {name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_shape_param(value: Any, expected: tuple[int, ...], name: str) -> None:
    try:
        actual = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{REPRO_ID} {name} must be a shape sequence, got {value!r}") from exc
    if actual != expected:
        raise ValueError(f"{REPRO_ID} {name} expects {expected}, got {actual}")


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 18:
        raise ValueError(f"{REPRO_ID} expects 18 inputs, got {len(inputs)}")

    table = _validate_tensor(inputs[0], (TABLE_ROWS, ROTARY_DIM), torch.float32, "arg304_1")
    positions = _validate_tensor(inputs[1], (BATCH, SEQ), torch.int64, "unsqueeze")
    mm_108 = _validate_tensor(inputs[2], (SEQ, HEADS * HEAD_DIM), torch.float32, "mm_108")
    mm_109 = _validate_tensor(inputs[3], (SEQ, HEADS * HEAD_DIM), torch.float32, "mm_109")

    devices = {table.device, positions.device, mm_108.device, mm_109.device}
    if len(devices) != 1:
        raise ValueError(f"{REPRO_ID} all tensor inputs must be on the same CUDA device")

    for index, expected in enumerate(SHAPE_PARAMS, start=4):
        _validate_shape_param(inputs[index], expected, f"_shape_param_{index - 4}")

    return table, positions, mm_108, mm_109


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _gptj_rope_gather_layout_kernel(
        table_ptr,
        pos_ptr,
        mm108_ptr,
        mm109_ptr,
        out108_ptr,
        out109_ptr,
        N_ROWS: tl.constexpr,
        HEADS_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        ROTARY_DIM_: tl.constexpr,
        ROTARY_PAIRS_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)

        row_mask = rows < N_ROWS
        elem_mask = row_mask[:, None] & (dims[None, :] < HEAD_DIM_)

        seq = rows // HEADS_
        head = rows - seq * HEADS_
        base = seq[:, None] * (HEADS_ * HEAD_DIM_) + head[:, None] * HEAD_DIM_
        offsets = base + dims[None, :]

        rotary = dims < ROTARY_DIM_
        rotary_mask = row_mask[:, None] & rotary[None, :]
        pair = dims // 2
        paired_dim = tl.where((dims & 1) == 0, dims + 1, dims - 1)
        rotate_sign = tl.where((dims & 1) == 0, -1.0, 1.0)

        table_row = tl.load(pos_ptr + seq, mask=row_mask, other=0)
        coeff_rotate = tl.load(
            table_ptr + table_row[:, None] * ROTARY_DIM_ + pair[None, :],
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        coeff_base = tl.load(
            table_ptr + table_row[:, None] * ROTARY_DIM_ + ROTARY_PAIRS_ + pair[None, :],
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)

        x108 = tl.load(mm108_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        x108_pair = tl.load(
            mm108_ptr + base + paired_dim[None, :],
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        y108_rotary = x108 * coeff_base + x108_pair * rotate_sign[None, :] * coeff_rotate
        y108 = tl.where(rotary[None, :], y108_rotary, x108)
        tl.store(out108_ptr + offsets, y108, mask=elem_mask)

        x109 = tl.load(mm109_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        x109_pair = tl.load(
            mm109_ptr + base + paired_dim[None, :],
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        y109_rotary = x109 * coeff_base + x109_pair * rotate_sign[None, :] * coeff_rotate
        y109 = tl.where(rotary[None, :], y109_rotary, x109)
        tl.store(out109_ptr + offsets, y109, mask=elem_mask)


def oracle_forward(inputs):
    """Run the full Repro.forward computation with a fused Triton kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_gptj_rope_gather_layout.py")

    table, positions, mm_108, mm_109 = _validate_inputs(inputs)
    out108 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=mm_108.device, dtype=torch.float32)
    out109 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=mm_109.device, dtype=torch.float32)

    block_rows = 4
    block_d = 64
    grid = (triton.cdiv(ROWS, block_rows), triton.cdiv(HEAD_DIM, block_d))
    _gptj_rope_gather_layout_kernel[grid](
        table,
        positions,
        mm_108,
        mm_109,
        out108,
        out109,
        N_ROWS=ROWS,
        HEADS_=HEADS,
        HEAD_DIM_=HEAD_DIM,
        ROTARY_DIM_=ROTARY_DIM,
        ROTARY_PAIRS_=ROTARY_PAIRS,
        BLOCK_ROWS=block_rows,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return (out108, out109)


def _check_layout(outputs: tuple[torch.Tensor, ...]) -> bool:
    ok = True
    for index, output in enumerate(outputs):
        layout_ok = (
            tuple(output.shape) == OUT_SHAPE
            and tuple(output.stride()) == OUT_STRIDE
            and output.dtype == torch.float32
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
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
