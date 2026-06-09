"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the full GPT-J position gather, duplicated-pair coefficient expansion, rotate-half arithmetic for both mm/mm_1 branches, tail concat, final permuted output layout, and boolean position-difference side output into one Triton kernel, whereas Inductor currently schedules the gather/expand/clone coefficient setup, two sibling RoPE pointwise branches, cat assembly, and permute layout as separated generic pointwise/layout work; Inductor cannot do this today because scheduler fusion does not inline the fixed position-indexed coefficient gather and duplicated rotate-half subgraphs through the concat and final permuted output layout for both branches; the fix is SCHEDULER_FUSION: teach Inductor to recognize GPT-J interleaved RoPE rotate-half patterns and sink the coefficient gather plus cat/permute materialization into one multi-output scheduler group."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

SEQ = 128
HEADS = 16
HEAD_DIM = 256
ROTARY_DIM = 64
TABLE_DIM = 64
ROWS = SEQ * HEADS
INPUT_SHAPE = (SEQ, HEADS * HEAD_DIM)
TABLE_SHAPE = (2048, TABLE_DIM)
MASK_SHAPE = (1, SEQ)
OUT_SHAPE = (1, HEADS, SEQ, HEAD_DIM)
OUT_STRIDE = (SEQ * HEADS * HEAD_DIM, HEAD_DIM, HEADS * HEAD_DIM, 1)
SHAPE_PARAMS = (
    (1, 128, 4096),
    (1, 128, 4096),
    (1, 128, 16, 256),
    (1, 128, 16, 256),
    (1, 128, 1, 32, 2),
    (1, 128, 1, 64),
    (1, 128, 1, 32, 2),
    (1, 128, 1, 64),
    (1, 128, 16, 64),
    (1, 128, 16, 64),
)

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


def _validate_tensor(value: object, shape: tuple[int, ...], dtype: torch.dtype, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} {name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{REPRO_ID} {name} expects shape {shape}, got {tuple(value.shape)}")
    if value.dtype is not dtype:
        raise ValueError(f"{REPRO_ID} {name} expects {dtype}, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{REPRO_ID} {name} expects a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{REPRO_ID} {name} expects contiguous input, got stride={value.stride()}")
    return value


def _validate_shape_param(value: object, expected: tuple[int, ...], name: str) -> None:
    try:
        actual = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{REPRO_ID} {name} must be a shape sequence, got {value!r}") from exc
    if actual != expected:
        raise ValueError(f"{REPRO_ID} {name} expects {expected}, got {actual}")


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _gptj_rope_layout_kernel(
        mm_ptr,
        mm_1_ptr,
        table_ptr,
        mask_ptr,
        out_mm_1_ptr,
        out_mm_ptr,
        N_ROWS: tl.constexpr,
        HEADS_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        ROTARY_DIM_: tl.constexpr,
        TABLE_DIM_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        row_mask = rows < N_ROWS
        elem_mask = row_mask[:, None] & (dims[None, :] < HEAD_DIM_)

        pos = rows // HEADS_
        head = rows - pos * HEADS_
        offsets = pos[:, None] * (HEADS_ * HEAD_DIM_) + head[:, None] * HEAD_DIM_ + dims[None, :]

        rotary = dims < ROTARY_DIM_
        pair_dims = tl.where((dims & 1) == 0, dims + 1, dims - 1)
        pair_offsets = (
            pos[:, None] * (HEADS_ * HEAD_DIM_)
            + head[:, None] * HEAD_DIM_
            + pair_dims[None, :]
        )
        rotary_mask = row_mask[:, None] & rotary[None, :]

        coeff_index = dims // 2
        coeff_offsets = pos[:, None] * TABLE_DIM_ + coeff_index[None, :]
        coeff_rotate = tl.load(table_ptr + coeff_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
        coeff_base = tl.load(table_ptr + coeff_offsets + 32, mask=rotary_mask, other=0.0).to(tl.float32)
        rotate_sign = tl.where((dims & 1) == 0, -1.0, 1.0)

        x_mm_1 = tl.load(mm_1_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        x_mm_1_pair = tl.load(mm_1_ptr + pair_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
        y_mm_1_rotary = x_mm_1 * coeff_base + x_mm_1_pair * rotate_sign[None, :] * coeff_rotate
        y_mm_1 = tl.where(rotary[None, :], y_mm_1_rotary, x_mm_1)
        tl.store(out_mm_1_ptr + offsets, y_mm_1, mask=elem_mask)

        x_mm = tl.load(mm_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        x_mm_pair = tl.load(mm_ptr + pair_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
        y_mm_rotary = x_mm * coeff_base + x_mm_pair * rotate_sign[None, :] * coeff_rotate
        y_mm = tl.where(rotary[None, :], y_mm_rotary, x_mm)
        tl.store(out_mm_ptr + offsets, y_mm, mask=elem_mask)

        first_dim_block = tl.program_id(1) == 0
        tl.store(mask_ptr + pos, rows != rows, mask=row_mask & (head == 0) & first_dim_block)


@oracle_impl(hardware="H100", shapes="(T([128, 4096], f32), T([128, 4096], f32), T([2048, 64], f32), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 16, 256]), S([1, 128, 16, 256]), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 16, 64]), S([1, 128, 16, 64]))")
def oracle_forward(inputs):
    """Run the full GPT-J RoPE layout repro scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_gptj_rope_layout.py")
    if len(inputs) != 13:
        raise ValueError(f"{REPRO_ID} expects 13 inputs, got {len(inputs)}")

    mm = _validate_tensor(inputs[0], INPUT_SHAPE, torch.float32, "mm")
    mm_1 = _validate_tensor(inputs[1], INPUT_SHAPE, torch.float32, "mm_1")
    table = _validate_tensor(inputs[2], TABLE_SHAPE, torch.float32, "arg7_1")
    for index, expected in enumerate(SHAPE_PARAMS, start=3):
        _validate_shape_param(inputs[index], expected, f"_shape_param_{index - 3}")

    mask = torch.empty(MASK_SHAPE, device=mm.device, dtype=torch.bool)
    out_mm_1 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=mm_1.device, dtype=torch.float32)
    out_mm = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=mm.device, dtype=torch.float32)

    block_rows = 4
    block_d = 64
    grid = (triton.cdiv(ROWS, block_rows), triton.cdiv(HEAD_DIM, block_d))
    _gptj_rope_layout_kernel[grid](
        mm,
        mm_1,
        table,
        mask,
        out_mm_1,
        out_mm,
        N_ROWS=ROWS,
        HEADS_=HEADS,
        HEAD_DIM_=HEAD_DIM,
        ROTARY_DIM_=ROTARY_DIM,
        TABLE_DIM_=TABLE_DIM,
        BLOCK_ROWS=block_rows,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return (mask, out_mm_1, out_mm)


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
