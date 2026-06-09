"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full two-output graph by materializing the scaled ALBERT attention layout clone while also producing its 128-row partial column sums, fusing the eleven sibling matrix column reductions into one multi-input partial-reduction kernel, and finalizing all twelve reductions with the captured left-associated add order, whereas Inductor emits one layout clone kernel, twelve separate partial reduction kernels, and a final add kernel; Inductor cannot do this today because its scheduler does not form a coordinated full-scope plan for a required layout side output plus many compatible sibling reductions; the fix is SCHEDULER_FUSION: teach reduction scheduling to share the row-tiled partial-reduction pass across sibling inputs and layout-producing side outputs when the final reduction/add tree is compatible."""
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
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


N = 4096
BMM_DIM0 = 512
LAYOUT_BATCH = 8
LAYOUT_HEADS = 64
LAYOUT_COL_GROUP = 64
LAYOUT_LAST = 512
ROWS_PER_PARTIAL = 128
N_PARTIALS = 32
N_SOURCES = 12
SCALE = 0.3535533905932738
PARTIAL_STRIDE = N_PARTIALS * N
DEFAULT_BLOCK_COLS = 32

if triton is not None:

    @triton.jit
    def _layout_clone_and_partial_kernel(
        bmm_ptr,
        base_ptr,
        partials_ptr,
        BLOCK_COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        row_block = tl.program_id(1)

        cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[:, None]
        rows_in_block = tl.arange(0, BLOCK_ROWS)[None, :]
        rows = row_block * BLOCK_ROWS + rows_in_block

        # Mirrors Inductor's lowered layout:
        # clone[row, col] = bmm[row % 512 + 512 * col + 2097152 * (row // 512)] * scale.
        bmm_offsets = (rows % 512) + 512 * cols + 2097152 * (rows // 512)
        values = tl.load(bmm_ptr + bmm_offsets, eviction_policy="evict_first")
        scale = tl.full([1, 1], 0.3535533905932738, tl.float32)
        scaled = values * scale

        tl.store(base_ptr + rows * 4096 + cols, scaled)

        partial = tl.sum(scaled, axis=1)
        out_cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        tl.store(
            partials_ptr + 11 * 131072 + row_block * 4096 + out_cols,
            partial,
        )

    @triton.jit
    def _eleven_matrix_partials_kernel(
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        x9,
        x10,
        partials_ptr,
        BLOCK_COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        row_block = tl.program_id(1)

        cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[:, None]
        rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[None, :]
        offsets = rows * 4096 + cols
        out_cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        partial_base = row_block * 4096 + out_cols

        values = tl.load(x0 + offsets, eviction_policy="evict_first")
        tl.store(partials_ptr + 0 * 131072 + partial_base, tl.sum(values, axis=1))

        values = tl.load(x1 + offsets, eviction_policy="evict_first")
        tl.store(partials_ptr + 1 * 131072 + partial_base, tl.sum(values, axis=1))

        values = tl.load(x2 + offsets, eviction_policy="evict_first")
        tl.store(partials_ptr + 2 * 131072 + partial_base, tl.sum(values, axis=1))

        values = tl.load(x3 + offsets, eviction_policy="evict_first")
        tl.store(partials_ptr + 3 * 131072 + partial_base, tl.sum(values, axis=1))

        values = tl.load(x4 + offsets, eviction_policy="evict_first")
        tl.store(partials_ptr + 4 * 131072 + partial_base, tl.sum(values, axis=1))

        values = tl.load(x5 + offsets, eviction_policy="evict_first")
        tl.store(partials_ptr + 5 * 131072 + partial_base, tl.sum(values, axis=1))

        values = tl.load(x6 + offsets, eviction_policy="evict_first")
        tl.store(partials_ptr + 6 * 131072 + partial_base, tl.sum(values, axis=1))

        values = tl.load(x7 + offsets, eviction_policy="evict_first")
        tl.store(partials_ptr + 7 * 131072 + partial_base, tl.sum(values, axis=1))

        values = tl.load(x8 + offsets, eviction_policy="evict_first")
        tl.store(partials_ptr + 8 * 131072 + partial_base, tl.sum(values, axis=1))

        values = tl.load(x9 + offsets, eviction_policy="evict_first")
        tl.store(partials_ptr + 9 * 131072 + partial_base, tl.sum(values, axis=1))

        values = tl.load(x10 + offsets, eviction_policy="evict_first")
        tl.store(partials_ptr + 10 * 131072 + partial_base, tl.sum(values, axis=1))

    @triton.jit
    def _finalize_partials_kernel(
        partials_ptr,
        out_ptr,
        BLOCK_COLS: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[:, None]
        chunks = tl.arange(0, 32)[None, :]
        offsets = chunks * 4096 + cols

        s0 = tl.sum(tl.load(partials_ptr + 0 * 131072 + offsets), axis=1)
        s1 = tl.sum(tl.load(partials_ptr + 1 * 131072 + offsets), axis=1)
        s2 = tl.sum(tl.load(partials_ptr + 2 * 131072 + offsets), axis=1)
        s3 = tl.sum(tl.load(partials_ptr + 3 * 131072 + offsets), axis=1)
        s4 = tl.sum(tl.load(partials_ptr + 4 * 131072 + offsets), axis=1)
        s5 = tl.sum(tl.load(partials_ptr + 5 * 131072 + offsets), axis=1)
        s6 = tl.sum(tl.load(partials_ptr + 6 * 131072 + offsets), axis=1)
        s7 = tl.sum(tl.load(partials_ptr + 7 * 131072 + offsets), axis=1)
        s8 = tl.sum(tl.load(partials_ptr + 8 * 131072 + offsets), axis=1)
        s9 = tl.sum(tl.load(partials_ptr + 9 * 131072 + offsets), axis=1)
        s10 = tl.sum(tl.load(partials_ptr + 10 * 131072 + offsets), axis=1)
        s11 = tl.sum(tl.load(partials_ptr + 11 * 131072 + offsets), axis=1)

        # Preserve the captured left-associated add order:
        # (((sum0 + sum1) + ... + sum10) + sum_clone).
        result = s0 + s1
        result = result + s2
        result = result + s3
        result = result + s4
        result = result + s5
        result = result + s6
        result = result + s7
        result = result + s8
        result = result + s9
        result = result + s10
        result = result + s11

        out_cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        tl.store(out_ptr + out_cols, result)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 27:
        raise ValueError(f"{REPRO_ID} expects 27 inputs, got {len(inputs)}")

    tensors = tuple(inputs[:12])
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("the first twelve repro inputs must be tensors")

    expected_matrix = (N, N)
    for index, tensor in enumerate(tensors[:11]):
        if tuple(tensor.shape) != expected_matrix:
            raise ValueError(
                f"input {index} expected shape={expected_matrix}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise TypeError(f"input {index} expected dtype torch.float32, got {tensor.dtype}")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA tensor inputs")
        if not tensor.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")

    bmm = tensors[11]
    expected_bmm = (BMM_DIM0, LAYOUT_COL_GROUP, LAYOUT_LAST)
    if tuple(bmm.shape) != expected_bmm:
        raise ValueError(f"input 11 expected shape={expected_bmm}, got {tuple(bmm.shape)}")
    if bmm.dtype != torch.float32:
        raise TypeError(f"input 11 expected dtype torch.float32, got {bmm.dtype}")
    if bmm.device != tensors[0].device:
        raise ValueError(f"input 11 is on {bmm.device}, expected {tensors[0].device}")
    if not bmm.is_contiguous():
        raise ValueError("input 11 must be contiguous")

    expected_shapes = (
        (N,),
        (N,),
        (N,),
        (N,),
        (N,),
        (N,),
        (N,),
        (N,),
        (N,),
        (N,),
        (N,),
        (LAYOUT_BATCH, LAYOUT_HEADS, LAYOUT_COL_GROUP, LAYOUT_LAST),
        (LAYOUT_BATCH, LAYOUT_LAST, N),
        (N, N),
        (N,),
    )
    actual_shapes = tuple(_shape_tuple(value) for value in inputs[12:])
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"shape parameters do not match repro: actual={actual_shapes}, "
            f"expected={expected_shapes}"
        )

    return tensors  # type: ignore[return-value]


@oracle_impl(hardware="H100", shapes="(T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([512, 64, 512], f32), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([8, 64, 64, 512]), S([8, 512, 4096]), S([4096, 4096]), S([4096]))")
def oracle_forward(inputs):
    """Run the full-scope oracle for Repro.forward()."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        view_34,
        view_64,
        view_94,
        view_124,
        view_154,
        view_184,
        view_214,
        view_244,
        view_274,
        view_304,
        view_334,
        bmm_46,
    ) = _validate_inputs(inputs)

    base = torch.empty_strided(
        (N, N),
        (N, 1),
        device=bmm_46.device,
        dtype=torch.float32,
    )
    partials = torch.empty_strided(
        (N_SOURCES, N_PARTIALS, N),
        (PARTIAL_STRIDE, N, 1),
        device=bmm_46.device,
        dtype=torch.float32,
    )
    out_sum = torch.empty_strided((N,), (1,), device=bmm_46.device, dtype=torch.float32)

    block_cols = DEFAULT_BLOCK_COLS
    grid = (triton.cdiv(N, block_cols), N_PARTIALS)
    _layout_clone_and_partial_kernel[grid](
        bmm_46,
        base,
        partials,
        BLOCK_COLS=block_cols,
        BLOCK_ROWS=ROWS_PER_PARTIAL,
        num_warps=8,
    )
    _eleven_matrix_partials_kernel[grid](
        view_34,
        view_64,
        view_94,
        view_124,
        view_154,
        view_184,
        view_214,
        view_244,
        view_274,
        view_304,
        view_334,
        partials,
        BLOCK_COLS=block_cols,
        BLOCK_ROWS=ROWS_PER_PARTIAL,
        num_warps=8,
    )
    _finalize_partials_kernel[(triton.cdiv(N, block_cols),)](
        partials,
        out_sum,
        BLOCK_COLS=block_cols,
        num_warps=4,
    )

    return base.permute(1, 0), out_sum


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
