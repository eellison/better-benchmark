"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Swin attention backward return tuple by reducing the two duplicate-index relative-position `index_put(accumulate=True)` outputs directly into their final `[169, 8]` buffers while the softmax-gradient producer writes the required `[16384, 49, 49]` tensor, whereas Inductor currently lowers the three large reductions, permutes/reshapes, softmax-gradient pointwise work, and duplicate-index scatters as separate generic scheduled regions; Inductor cannot do this today because scheduler/codegen does not recognize this multi-output rowwise softmax-backward producer feeding a structured duplicate-index scatter-reduce epilogue; the fix is SCATTER_REDUCE: add a lowering that keeps the rowwise reduction producer in registers and accumulates duplicate relative-position rows directly into the final dense outputs."""
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

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


BATCH = 2048
HEADS = 8
TOKENS = 49
REL_POS = 169
OUT3_ROWS = BATCH * HEADS
OUT3_SHAPE = (OUT3_ROWS, TOKENS, TOKENS)
OUT3_STRIDE = (TOKENS * TOKENS, TOKENS, 1)
SCATTER_SHAPE = (REL_POS, HEADS)
SCATTER_STRIDE = (HEADS, 1)

BLOCK_N = 128
BLOCK_B = 64
ZERO_BLOCK = 2048


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _zero_pair_kernel(
        out0_ptr,
        out1_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < n_elements
        zeros = tl.zeros((BLOCK,), tl.float32)
        tl.store(out0_ptr + offsets, zeros, mask=active)
        tl.store(out1_ptr + offsets, zeros, mask=active)

    @triton.jit
    def _sum_scatter_kernel(
        fma20_ptr,
        index_ptr,
        out_ptr,
        fma_s0: tl.constexpr,
        fma_s1: tl.constexpr,
        fma_s2: tl.constexpr,
        fma_s3: tl.constexpr,
        index_s0: tl.constexpr,
        index_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        BATCH_: tl.constexpr,
        TOKENS_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_B_: tl.constexpr,
    ):
        n_block = tl.program_id(0)
        head = tl.program_id(1)
        row = tl.program_id(2)

        ns = n_block * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        cols = tl.arange(0, BLOCK_B_)
        n_mask = ns < BATCH_
        col_mask = cols < TOKENS_
        mask = n_mask[:, None] & col_mask[None, :]

        values = tl.load(
            fma20_ptr
            + ns[:, None] * fma_s0
            + head * fma_s1
            + row * fma_s2
            + cols[None, :] * fma_s3,
            mask=mask,
            other=0.0,
        )
        partial = tl.sum(values, axis=0)
        rel = tl.load(
            index_ptr + row * index_s0 + cols * index_s1,
            mask=col_mask,
            other=0,
        ).to(tl.int64)
        tl.atomic_add(
            out_ptr + rel * out_s0 + head * out_s1,
            partial,
            sem="relaxed",
            mask=col_mask,
        )

    @triton.jit
    def _softmax_backward_store_scatter_kernel(
        bmm_ptr,
        div_ptr,
        index_ptr,
        out_scatter_ptr,
        out3_ptr,
        bmm_s0: tl.constexpr,
        bmm_s1: tl.constexpr,
        bmm_s2: tl.constexpr,
        div_s0: tl.constexpr,
        div_s1: tl.constexpr,
        div_s2: tl.constexpr,
        div_s3: tl.constexpr,
        index_s0: tl.constexpr,
        index_s1: tl.constexpr,
        scatter_s0: tl.constexpr,
        scatter_s1: tl.constexpr,
        out3_s0: tl.constexpr,
        out3_s1: tl.constexpr,
        out3_s2: tl.constexpr,
        BATCH_: tl.constexpr,
        HEADS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_B_: tl.constexpr,
    ):
        n_block = tl.program_id(0)
        head = tl.program_id(1)
        row = tl.program_id(2)

        ns = n_block * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        cols = tl.arange(0, BLOCK_B_)
        n_mask = ns < BATCH_
        col_mask = cols < TOKENS_
        mask = n_mask[:, None] & col_mask[None, :]
        logical_rows = ns * HEADS_ + head

        bmm = tl.load(
            bmm_ptr
            + logical_rows[:, None] * bmm_s0
            + row * bmm_s1
            + cols[None, :] * bmm_s2,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        div = tl.load(
            div_ptr
            + ns[:, None] * div_s0
            + head * div_s1
            + row * div_s2
            + cols[None, :] * div_s3,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        mul = bmm * div
        row_sum = tl.sum(mul, axis=1)
        grad = (-div) * row_sum[:, None] + mul

        tl.store(
            out3_ptr
            + logical_rows[:, None] * out3_s0
            + row * out3_s1
            + cols[None, :] * out3_s2,
            grad,
            mask=mask,
        )

        partial = tl.sum(grad, axis=0)
        rel = tl.load(
            index_ptr + row * index_s0 + cols * index_s1,
            mask=col_mask,
            other=0,
        ).to(tl.int64)
        tl.atomic_add(
            out_scatter_ptr + rel * scatter_s0 + head * scatter_s1,
            partial,
            sem="relaxed",
            mask=col_mask,
        )


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} expected shape={shape}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{name} expected dtype={dtype}, got {value.dtype}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 9:
        raise ValueError(f"expected 9 inputs, got {len(inputs)}")

    fma_20, primals_58, bmm_133, div_4, primals_43, sp0, sp1, sp2, sp3 = inputs
    fma_20 = _require_tensor("fma_20", fma_20, (BATCH, HEADS, TOKENS, TOKENS), torch.float32)
    primals_58 = _require_tensor("primals_58", primals_58, (TOKENS, TOKENS), torch.int64)
    bmm_133 = _require_tensor("bmm_133", bmm_133, OUT3_SHAPE, torch.float32)
    div_4 = _require_tensor("div_4", div_4, (BATCH, HEADS, TOKENS, TOKENS), torch.float32)
    primals_43 = _require_tensor("primals_43", primals_43, (TOKENS, TOKENS), torch.int64)

    if list(sp0) != [TOKENS * TOKENS, HEADS]:
        raise ValueError(f"unexpected _shape_param_0: {sp0}")
    if list(sp1) != [BATCH, HEADS, TOKENS, TOKENS]:
        raise ValueError(f"unexpected _shape_param_1: {sp1}")
    if list(sp2) != [TOKENS * TOKENS, HEADS]:
        raise ValueError(f"unexpected _shape_param_2: {sp2}")
    if list(sp3) != [OUT3_ROWS, TOKENS, TOKENS]:
        raise ValueError(f"unexpected _shape_param_3: {sp3}")

    return fma_20, primals_58, bmm_133, div_4, primals_43


@oracle_impl(hardware="H100", shapes="(T([2048, 8, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([16384, 49, 49], f32), T([2048, 8, 49, 49], f32, stride=(19456, 2432, 49, 1)), T([49, 49], i64, gen=Index(169)), S([2401, 8]), S([2048, 8, 49, 49]), S([2401, 8]), S([16384, 49, 49]))")
def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full repro scope with exact output count, shapes, dtypes, and strides."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")

    fma_20, primals_58, bmm_133, div_4, primals_43 = _validate_inputs(inputs)

    out0 = torch.empty_strided(
        SCATTER_SHAPE,
        SCATTER_STRIDE,
        device=fma_20.device,
        dtype=torch.float32,
    )
    out1 = torch.empty_strided(
        SCATTER_SHAPE,
        SCATTER_STRIDE,
        device=fma_20.device,
        dtype=torch.float32,
    )
    out3 = torch.empty_strided(
        OUT3_SHAPE,
        OUT3_STRIDE,
        device=fma_20.device,
        dtype=torch.float32,
    )

    scatter_elements = REL_POS * HEADS
    _zero_pair_kernel[(triton.cdiv(scatter_elements, ZERO_BLOCK),)](
        out0,
        out1,
        scatter_elements,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )
    grid = (triton.cdiv(BATCH, BLOCK_N), HEADS, TOKENS)
    _sum_scatter_kernel[grid](
        fma_20,
        primals_58,
        out0,
        fma_s0=int(fma_20.stride(0)),
        fma_s1=int(fma_20.stride(1)),
        fma_s2=int(fma_20.stride(2)),
        fma_s3=int(fma_20.stride(3)),
        index_s0=int(primals_58.stride(0)),
        index_s1=int(primals_58.stride(1)),
        out_s0=int(out0.stride(0)),
        out_s1=int(out0.stride(1)),
        BATCH_=BATCH,
        TOKENS_=TOKENS,
        BLOCK_N_=BLOCK_N,
        BLOCK_B_=BLOCK_B,
        num_warps=8,
    )
    _softmax_backward_store_scatter_kernel[grid](
        bmm_133,
        div_4,
        primals_43,
        out1,
        out3,
        bmm_s0=int(bmm_133.stride(0)),
        bmm_s1=int(bmm_133.stride(1)),
        bmm_s2=int(bmm_133.stride(2)),
        div_s0=int(div_4.stride(0)),
        div_s1=int(div_4.stride(1)),
        div_s2=int(div_4.stride(2)),
        div_s3=int(div_4.stride(3)),
        index_s0=int(primals_43.stride(0)),
        index_s1=int(primals_43.stride(1)),
        scatter_s0=int(out1.stride(0)),
        scatter_s1=int(out1.stride(1)),
        out3_s0=int(out3.stride(0)),
        out3_s1=int(out3.stride(1)),
        out3_s2=int(out3.stride(2)),
        BATCH_=BATCH,
        HEADS_=HEADS,
        TOKENS_=TOKENS,
        BLOCK_N_=BLOCK_N,
        BLOCK_B_=BLOCK_B,
        num_warps=8,
    )
    return out0, out1, out3


# --- CLI entry point ---
def main() -> None:
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
