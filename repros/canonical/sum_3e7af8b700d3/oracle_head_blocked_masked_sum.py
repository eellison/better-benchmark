"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete masked dropout-weighted inner reduction and dependent pointwise epilogue while tiling the broadcast head dimension so the shared mask is loaded once per head block, whereas Inductor flattens the non-reduction dimensions into independent rows and reloads the broadcast mask for every head; Inductor cannot do this today because its reduction scheduler does not form head-blocked row programs around broadcasted non-reduction operands; the fix is NEW_PATTERN: add a small-inner-reduction template that tiles broadcast dimensions with the row reduction and keeps the dependent FMA epilogue fused."""
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


BATCH = 128
HEADS = 12
ROWS_PER_HEAD = 128
COLS = 128
HEAD_BLOCK = 16


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


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

    @triton.jit
    def _head_blocked_sum_kernel(
        bmm_ptr,
        dropout_ptr,
        grad_ptr,
        mask_ptr,
        fill_ptr,
        out_ptr,
        HEAD_BLOCK_: tl.constexpr,
        HEADS_: tl.constexpr,
        ROWS_PER_HEAD_: tl.constexpr,
        COL_BLOCK_: tl.constexpr,
    ):
        bm = tl.program_id(0)
        head_block = tl.program_id(1)

        batch = bm // ROWS_PER_HEAD_
        row = bm - batch * ROWS_PER_HEAD_
        heads = head_block * HEAD_BLOCK_ + tl.arange(0, HEAD_BLOCK_)
        cols = tl.arange(0, COL_BLOCK_)
        valid_heads = heads < HEADS_

        offsets = (
            batch * (HEADS_ * ROWS_PER_HEAD_ * COL_BLOCK_)
            + heads[:, None] * (ROWS_PER_HEAD_ * COL_BLOCK_)
            + row * COL_BLOCK_
            + cols[None, :]
        )
        head_mask = valid_heads[:, None]

        bmm = tl.load(bmm_ptr + offsets, mask=head_mask, other=0.0)
        keep = tl.load(dropout_ptr + offsets, mask=head_mask, other=0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=head_mask, other=0.0)

        scaled_keep = keep * 1.1111111111111112
        weighted = bmm * scaled_keep
        product = weighted * grad
        row_sum = tl.sum(product, axis=1)[:, None].to(tl.float32)

        shared_mask_offsets = batch * (ROWS_PER_HEAD_ * COL_BLOCK_) + row * COL_BLOCK_ + cols
        shared_mask = tl.load(mask_ptr + shared_mask_offsets).to(tl.int64) == 0
        fill = tl.load(fill_ptr + 0)

        epilogue = _fma_rn_f32(-grad, row_sum, product)
        selected = tl.where(shared_mask[None, :], fill, epilogue)
        result = selected * 0.125
        tl.store(out_ptr + offsets, result, mask=head_mask)


def _validate_inputs(
    bmm_1: torch.Tensor,
    arg273_1: torch.Tensor,
    arg272_1: torch.Tensor,
    arg100_1: torch.Tensor,
    full_1: torch.Tensor,
    shape_param_0: object,
    shape_param_1: object,
) -> None:
    if triton is None:
        raise RuntimeError("triton is not available")
    if bmm_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if tuple(shape_param_0) != (BATCH, HEADS, ROWS_PER_HEAD, COLS):
        raise ValueError(f"unexpected view shape: {shape_param_0}")
    if tuple(shape_param_1) != (BATCH * HEADS, ROWS_PER_HEAD, COLS):
        raise ValueError(f"unexpected output shape: {shape_param_1}")
    if bmm_1.dtype != torch.float32 or arg272_1.dtype != torch.float32:
        raise ValueError("expected float32 data tensors")
    if arg273_1.dtype != torch.bool or arg100_1.dtype != torch.bool:
        raise ValueError("expected bool mask tensors")
    if full_1.dtype != torch.float32 or tuple(full_1.shape) != ():
        raise ValueError("expected float32 scalar fill tensor")
    if tuple(bmm_1.shape) != (BATCH * HEADS, ROWS_PER_HEAD, COLS):
        raise ValueError(f"unexpected bmm shape: {tuple(bmm_1.shape)}")
    if tuple(arg273_1.shape) != (BATCH, HEADS, ROWS_PER_HEAD, COLS):
        raise ValueError(f"unexpected dropout mask shape: {tuple(arg273_1.shape)}")
    if tuple(arg272_1.shape) != (BATCH, HEADS, ROWS_PER_HEAD, COLS):
        raise ValueError(f"unexpected grad shape: {tuple(arg272_1.shape)}")
    if tuple(arg100_1.shape) != (BATCH, 1, ROWS_PER_HEAD, COLS):
        raise ValueError(f"unexpected broadcast mask shape: {tuple(arg100_1.shape)}")
    expected_bmm_stride = (ROWS_PER_HEAD * COLS, COLS, 1)
    expected_4d_stride = (HEADS * ROWS_PER_HEAD * COLS, ROWS_PER_HEAD * COLS, COLS, 1)
    expected_mask_stride = (ROWS_PER_HEAD * COLS, ROWS_PER_HEAD * COLS, COLS, 1)
    if bmm_1.stride() != expected_bmm_stride:
        raise ValueError(f"unexpected bmm stride: {bmm_1.stride()}")
    if arg273_1.stride() != expected_4d_stride or arg272_1.stride() != expected_4d_stride:
        raise ValueError("unexpected contiguous 4D tensor stride")
    if arg100_1.stride() != expected_mask_stride:
        raise ValueError(f"unexpected broadcast mask stride: {arg100_1.stride()}")


def oracle_forward(inputs):
    """Run the full Repro.forward computation."""
    (
        bmm_1,
        arg273_1,
        arg272_1,
        arg100_1,
        full_1,
        shape_param_0,
        shape_param_1,
    ) = inputs
    _validate_inputs(
        bmm_1,
        arg273_1,
        arg272_1,
        arg100_1,
        full_1,
        shape_param_0,
        shape_param_1,
    )

    out = torch.empty_strided(
        (BATCH * HEADS, ROWS_PER_HEAD, COLS),
        (ROWS_PER_HEAD * COLS, COLS, 1),
        device=bmm_1.device,
        dtype=torch.float32,
    )
    grid = (BATCH * ROWS_PER_HEAD, triton.cdiv(HEADS, HEAD_BLOCK))
    _head_blocked_sum_kernel[grid](
        bmm_1,
        arg273_1,
        arg272_1,
        arg100_1,
        full_1,
        out,
        HEAD_BLOCK_=HEAD_BLOCK,
        HEADS_=HEADS,
        ROWS_PER_HEAD_=ROWS_PER_HEAD,
        COL_BLOCK_=COLS,
        num_warps=4,
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
