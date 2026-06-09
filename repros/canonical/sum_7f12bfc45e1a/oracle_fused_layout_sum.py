"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete cat/view/permute/clone/view/transpose plus column-sum scope by writing the returned transpose-view backing buffer and accumulating per-column reduction partials from the original strided inputs in the same Triton pass, whereas Inductor currently materializes the cloned layout and then rereads that full buffer in separate reduction kernels; Inductor cannot do this today because its scheduler treats the returned clone as a materialization barrier for the sibling reduction consumer; the fix is SCHEDULER_FUSION: allow a layout materialization producer that is also returned to feed a same-pass reduction partial output before the final small partial reduction."""
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


ROWS_PER_PARTIAL = 373
EXPECTED_INPUT_SHAPE = (128, 12, 256, 64)
EXPECTED_INPUT_STRIDE = (196608, 64, 768, 1)
EXPECTED_SHAPE_PARAMS = (
    [2, 128, 12, 256, 64],
    [128, 256, 1536],
    [32768, 1536],
    [1536],
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _layout_and_partial_sum_kernel(
        in0_ptr,
        in1_ptr,
        layout_ptr,
        partial_ptr,
        ROWS_PER_PARTIAL_: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        partial_tile = tl.program_id(0)
        col_tile = tl.program_id(1)

        cols = col_tile * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < 1536
        source = cols // 768
        head_dim = cols - source * 768
        head = head_dim // 64
        dim = head_dim - head * 64

        row_lanes = tl.arange(0, BLOCK_R)
        acc = tl.zeros([BLOCK_C, BLOCK_R], tl.float32)

        for row_start in tl.range(0, ROWS_PER_PARTIAL_, BLOCK_R):
            rows_in_tile = row_start + row_lanes
            rows = partial_tile * ROWS_PER_PARTIAL_ + rows_in_tile
            row_mask = (rows_in_tile < ROWS_PER_PARTIAL_) & (rows < 32768)

            batch = rows // 256
            seq = rows - batch * 256
            input_offsets = (
                batch[None, :] * 196608
                + seq[None, :] * 768
                + head[:, None] * 64
                + dim[:, None]
            )
            full_mask = col_mask[:, None] & row_mask[None, :]

            values0 = tl.load(
                in0_ptr + input_offsets,
                mask=full_mask & (source[:, None] == 0),
                other=0.0,
            )
            values1 = tl.load(
                in1_ptr + input_offsets,
                mask=full_mask & (source[:, None] == 1),
                other=0.0,
            )
            values = tl.where(source[:, None] == 0, values0, values1)
            tl.store(
                layout_ptr + rows[None, :] * 1536 + cols[:, None],
                values,
                mask=full_mask,
            )
            acc = tl.where(row_mask[None, :], acc + values, acc)

        partial = tl.sum(acc, axis=1)
        tl.store(partial_ptr + partial_tile * 1536 + cols, partial, mask=col_mask)

    @triton.jit
    def _finish_sum_kernel(
        partial_ptr,
        sum_ptr,
        NUM_PARTIALS: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        col_tile = tl.program_id(0)
        cols = col_tile * BLOCK_C + tl.arange(0, BLOCK_C)
        lanes = tl.arange(0, BLOCK_T)
        col_mask = cols < 1536
        acc = tl.zeros([BLOCK_C, BLOCK_T], tl.float32)

        for tile_start in tl.range(0, NUM_PARTIALS, BLOCK_T):
            tiles = tile_start + lanes
            mask = col_mask[:, None] & (tiles[None, :] < NUM_PARTIALS)
            values = tl.load(
                partial_ptr + tiles[None, :] * 1536 + cols[:, None],
                mask=mask,
                other=0.0,
            )
            acc = tl.where(mask, acc + values, acc)

        sums = tl.sum(acc, axis=1)
        tl.store(sum_ptr + cols, sums, mask=col_mask)


def _require_shape_param(actual, expected, name: str) -> None:
    if list(actual) != expected:
        raise ValueError(f"unexpected {name}: got={actual}, expected={expected}")


def _validate_inputs(inputs) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 6:
        raise ValueError(f"expected 6 inputs, got {len(inputs)}")

    getitem_1, getitem_2, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(getitem_1, torch.Tensor) or not isinstance(getitem_2, torch.Tensor):
        raise TypeError("expected the first two inputs to be tensors")
    if getitem_1.device.type != "cuda" or getitem_2.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if getitem_1.dtype != torch.float32 or getitem_2.dtype != torch.float32:
        raise ValueError("expected fp32 tensor inputs")
    if tuple(getitem_1.shape) != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"unexpected first input shape: {tuple(getitem_1.shape)}")
    if tuple(getitem_2.shape) != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"unexpected second input shape: {tuple(getitem_2.shape)}")
    if tuple(getitem_1.stride()) != EXPECTED_INPUT_STRIDE:
        raise ValueError(f"unexpected first input stride: {tuple(getitem_1.stride())}")
    if tuple(getitem_2.stride()) != EXPECTED_INPUT_STRIDE:
        raise ValueError(f"unexpected second input stride: {tuple(getitem_2.stride())}")
    for actual, expected, name in zip(
        (shape0, shape1, shape2, shape3),
        EXPECTED_SHAPE_PARAMS,
        ("shape0", "shape1", "shape2", "shape3"),
    ):
        _require_shape_param(actual, expected, name)
    return getitem_1, getitem_2


@oracle_impl(hardware="H100", shapes="(T([128, 12, 256, 64], f32, stride=(196608, 64, 768, 1)), T([128, 12, 256, 64], f32, stride=(196608, 64, 768, 1)), S([2, 128, 12, 256, 64]), S([128, 256, 1536]), S([32768, 1536]), S([1536]))")
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
    getitem_1, getitem_2 = _validate_inputs(inputs)
    rows = 32768
    cols = 1536
    num_partials = (rows + ROWS_PER_PARTIAL - 1) // ROWS_PER_PARTIAL

    layout = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=getitem_1.device,
        dtype=torch.float32,
    )
    partials = torch.empty_strided(
        (num_partials, cols),
        (cols, 1),
        device=getitem_1.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided(
        (cols,),
        (1,),
        device=getitem_1.device,
        dtype=torch.float32,
    )

    block_r = 16
    block_c = 64
    grid = (num_partials, triton.cdiv(cols, block_c))
    _layout_and_partial_sum_kernel[grid](
        getitem_1,
        getitem_2,
        layout,
        partials,
        ROWS_PER_PARTIAL_=ROWS_PER_PARTIAL,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=1,
    )
    final_block_c = 32
    _finish_sum_kernel[(triton.cdiv(cols, final_block_c),)](
        partials,
        sum_out,
        NUM_PARTIALS=num_partials,
        BLOCK_T=32,
        BLOCK_C=final_block_c,
        num_warps=4,
        num_stages=1,
    )
    return layout.as_strided((cols, rows), (1, cols)), sum_out


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
