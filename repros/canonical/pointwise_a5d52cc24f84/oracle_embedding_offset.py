"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete `arg0_1 + 2` followed by `aten.embedding.default` Repro.forward scope as one Triton gather that applies the constant row offset in the table address and writes the required fresh contiguous output, whereas Inductor currently materializes the int64 add result before launching the embedding gather; Inductor cannot do this today because its algebraic simplifier does not fold affine integer index producers into embedding address generation before scheduler/codegen lowers the gather; the fix is ALGEBRAIC_ELIMINATION: add a guarded embedding-index affine rewrite that removes the intermediate index tensor and emits a direct offset gather for constant-add index producers."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"

ROWS = 4
DEFAULT_SEQ = 512
HIDDEN = 768
VOCAB = 2050
INDEX_OFFSET = 2


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=4),
            triton.Config({"BLOCK_M": 2, "BLOCK_N": 1024}, num_warps=4),
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=8),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        ],
        key=["seq_len"],
    )
    @triton.jit
    def _embedding_offset_kernel(
        ids_ptr,
        table_ptr,
        out_ptr,
        ids_stride0: tl.constexpr,
        ids_stride1: tl.constexpr,
        seq_len: tl.constexpr,
        n_tokens: tl.constexpr,
        hidden: tl.constexpr,
        table_stride0: tl.constexpr,
        table_stride1: tl.constexpr,
        index_offset: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        token_block = tl.program_id(0)
        col_block = tl.program_id(1)
        tokens = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
        batch = tokens // seq_len
        seq = tokens - batch * seq_len
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (tokens[:, None] < n_tokens) & (cols[None, :] < hidden)

        src_row = tl.load(
            ids_ptr + batch * ids_stride0 + seq * ids_stride1,
            mask=tokens < n_tokens,
            other=0,
        ) + index_offset
        values = tl.load(
            table_ptr + src_row[:, None] * table_stride0 + cols[None, :] * table_stride1,
            mask=mask,
            other=0.0,
        )
        tl.store(out_ptr + tokens[:, None] * hidden + cols[None, :], values, mask=mask)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_embedding_offset.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    arg0_1, arg1_1 = inputs
    if not isinstance(arg0_1, torch.Tensor) or not isinstance(arg1_1, torch.Tensor):
        raise TypeError("oracle inputs must be tensors")
    if arg0_1.device.type != "cuda" or arg1_1.device.type != "cuda":
        raise RuntimeError("oracle_embedding_offset.py expects CUDA inputs")
    if arg0_1.dtype != torch.int64:
        raise TypeError(f"expected int64 ids, got {arg0_1.dtype}")
    if arg1_1.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"expected fp16/fp32 embedding table, got {arg1_1.dtype}")
    if arg0_1.ndim != 2 or arg0_1.shape[0] != ROWS:
        raise ValueError(f"unexpected ids shape: {tuple(arg0_1.shape)}")
    if tuple(arg1_1.shape) != (VOCAB, HIDDEN):
        raise ValueError(f"unexpected embedding table shape: {tuple(arg1_1.shape)}")
    if not arg1_1.is_contiguous():
        raise ValueError("oracle expects the embedding table to be contiguous")
    return arg0_1, arg1_1


def oracle_embedding_offset(arg0_1: torch.Tensor, arg1_1: torch.Tensor) -> torch.Tensor:
    """Compute Repro.forward with a direct offset embedding gather."""
    seq_len = arg0_1.shape[1]
    out = torch.empty_strided(
        (ROWS, seq_len, HIDDEN),
        (seq_len * HIDDEN, HIDDEN, 1),
        device=arg1_1.device,
        dtype=arg1_1.dtype,
    )
    grid = lambda meta: (triton.cdiv(ROWS * seq_len, meta["BLOCK_M"]), triton.cdiv(HIDDEN, meta["BLOCK_N"]))
    _embedding_offset_kernel[grid](
        arg0_1,
        arg1_1,
        out,
        ids_stride0=arg0_1.stride(0),
        ids_stride1=arg0_1.stride(1),
        seq_len=seq_len,
        n_tokens=ROWS * seq_len,
        hidden=HIDDEN,
        table_stride0=arg1_1.stride(0),
        table_stride1=arg1_1.stride(1),
        index_offset=INDEX_OFFSET,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([4, 512], i64), T([2050, 768], f16))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full-scope oracle for Repro.forward."""
    arg0_1, arg1_1 = _validate_inputs(inputs)
    return oracle_embedding_offset(arg0_1, arg1_1)


def _check_output_layout(output: torch.Tensor, seq_len: int, dtype: torch.dtype) -> bool:
    expected_shape = (ROWS, seq_len, HIDDEN)
    expected_stride = (seq_len * HIDDEN, HIDDEN, 1)
    ok = (
        tuple(output.shape) == expected_shape
        and tuple(output.stride()) == expected_stride
        and output.dtype == dtype
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(output.shape)} stride={output.stride()} dtype={output.dtype})"
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
        ok = _check_output_layout(layout_out, inputs[0].shape[1], inputs[1].dtype) and ok
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
