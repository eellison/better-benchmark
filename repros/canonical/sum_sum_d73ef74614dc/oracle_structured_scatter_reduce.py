"""
Oracle for sum_sum_d73ef74614dc

Gap diagnosis:
  Classification: SCATTER_REDUCE
  What oracle does differently: Computes the full Qwen RMSNorm-backward-style
    row gather/mask/reduce and indexed scatter-add tuple with Triton, writing
    the required vocabulary-gradient output directly from source rows.
  What Inductor change would fix: Add a structured row-index scatter-reduce
    lowering that fuses the RMSNorm row reduction, side weight-gradient
    reduction, and duplicate-index scatter-add into the materialized output.
"""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info as _harness_get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


BATCH = 4
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 2048
VOCAB = 151936
OUT0_BLOCK_C = 8


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def get_hardware_info():
    """Return GPU info, tolerating PyTorch device-property name differences."""
    try:
        return _harness_get_hardware_info()
    except AttributeError:
        props = torch.cuda.get_device_properties(0)
        shared_mem = getattr(
            props,
            "max_shared_memory_per_multiprocessor",
            getattr(props, "shared_memory_per_multiprocessor", None),
        )
        return {
            "name": props.name,
            "sm_major": props.major,
            "sm_minor": props.minor,
            "num_sms": props.multi_processor_count,
            "shared_mem_per_sm": shared_mem,
            "total_mem_gb": props.total_memory / 1e9,
        }


if triton is not None:

    @triton.jit
    def _bf16_add3(a, b, c):
        partial = (a + b).to(tl.bfloat16).to(tl.float32)
        return (partial + c).to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _out0_weight_grad_kernel(
        mm37,
        mm39,
        mm41,
        arg48,
        arg49,
        out0,
        ROWS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        rows = tl.arange(0, ROWS_)
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < HIDDEN_
        offsets = rows[:, None] * HIDDEN_ + cols[None, :]

        mm_sum = _bf16_add3(
            tl.load(mm37 + offsets, mask=col_mask[None, :], other=0.0).to(tl.float32),
            tl.load(mm39 + offsets, mask=col_mask[None, :], other=0.0).to(tl.float32),
            tl.load(mm41 + offsets, mask=col_mask[None, :], other=0.0).to(tl.float32),
        )
        x = tl.load(arg48 + offsets, mask=col_mask[None, :], other=0.0).to(tl.float32)
        rstd = tl.load(arg49 + rows, mask=rows < ROWS_, other=0.0).to(tl.float32)
        normalized = (x * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        product = (mm_sum * normalized).to(tl.bfloat16).to(tl.float32)
        tl.store(out0 + cols, tl.sum(product, axis=0).to(tl.bfloat16), mask=col_mask)

    @triton.jit
    def _scatter_rows_kernel(
        mm37,
        mm39,
        mm41,
        gamma,
        arg48,
        arg49,
        add57,
        row_index,
        full_scalar,
        out1,
        ROWS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        VOCAB_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        offsets = row * HIDDEN_ + cols

        mm_sum = _bf16_add3(
            tl.load(mm37 + offsets).to(tl.float32),
            tl.load(mm39 + offsets).to(tl.float32),
            tl.load(mm41 + offsets).to(tl.float32),
        )
        token_grad = (
            mm_sum * tl.load(gamma + cols).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        x = tl.load(arg48 + offsets).to(tl.float32)
        rstd = tl.load(arg49 + row).to(tl.float32)
        row_dot = tl.sum(token_grad * x, axis=0)

        correction_scale = -0.5 * row_dot * rstd * rstd * rstd / HIDDEN_
        grad_x = token_grad * rstd + correction_scale * (x * 2.0)
        grad_x_bf16 = grad_x.to(tl.bfloat16).to(tl.float32)
        add_val = (
            tl.load(add57 + offsets).to(tl.float32) + grad_x_bf16
        ).to(tl.bfloat16).to(tl.float32)

        raw_dest = tl.load(row_index + row).to(tl.int64)
        dest = tl.where(raw_dest < 0, raw_dest + VOCAB_, raw_dest)
        full_val = tl.load(full_scalar).to(tl.float32)
        scatter_val = tl.where(raw_dest == -1, full_val, add_val).to(tl.bfloat16)
        valid = (dest >= 0) & (dest < VOCAB_)
        tl.atomic_add(
            out1 + dest * HIDDEN_ + cols,
            scatter_val,
            sem="relaxed",
            mask=valid,
        )


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 14:
        raise ValueError(f"expected 14 inputs, got {len(inputs)}")

    tensors = inputs[:9]
    if not all(isinstance(inp, torch.Tensor) for inp in tensors):
        raise ValueError("first nine inputs must be tensors")
    if not all(inp.is_cuda for inp in tensors):
        raise ValueError("this oracle expects CUDA inputs")
    if not all(inp.is_contiguous() for inp in tensors):
        raise ValueError("this oracle expects contiguous tensor inputs")

    expected = (
        (HIDDEN, HIDDEN),
        (HIDDEN, HIDDEN),
        (HIDDEN, HIDDEN),
        (HIDDEN,),
        (BATCH, SEQ, HIDDEN),
        (BATCH, SEQ, 1),
        (BATCH, SEQ, HIDDEN),
        (BATCH, SEQ),
        (),
    )
    actual = tuple(tuple(inp.shape) for inp in tensors)
    if actual != expected:
        raise ValueError(f"unsupported input shapes: expected {expected}, got {actual}")

    expected_dtypes = (
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
        torch.int64,
        torch.float32,
    )
    actual_dtypes = tuple(inp.dtype for inp in tensors)
    if actual_dtypes != expected_dtypes:
        raise ValueError(
            f"unsupported input dtypes: expected {expected_dtypes}, got {actual_dtypes}"
        )


def oracle_forward(inputs):
    """Run the full-scope oracle for Repro()(*make_inputs())."""
    _validate_inputs(inputs)
    (
        mm37,
        mm39,
        mm41,
        gamma,
        arg48,
        arg49,
        add57,
        row_index,
        full_scalar,
        *_shape_params,
    ) = inputs

    out0 = torch.empty((HIDDEN,), device=mm37.device, dtype=torch.bfloat16)
    out1 = torch.empty((VOCAB, HIDDEN), device=mm37.device, dtype=torch.bfloat16)
    out1.zero_()

    _out0_weight_grad_kernel[(triton.cdiv(HIDDEN, OUT0_BLOCK_C),)](
        mm37,
        mm39,
        mm41,
        arg48,
        arg49,
        out0,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        BLOCK_C=OUT0_BLOCK_C,
        num_warps=8,
    )
    _scatter_rows_kernel[(ROWS,)](
        mm37,
        mm39,
        mm41,
        gamma,
        arg48,
        arg49,
        add57,
        row_index,
        full_scalar,
        out1,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        VOCAB_=VOCAB,
        BLOCK_C=HIDDEN,
        num_warps=8,
    )
    return out0, out1


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify correctness against eager Repro"
    )
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument(
        "--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check"
    )
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=200, help="Benchmark repetitions")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument(
        "--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt"
    )
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info")
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
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward, REPRO_DIR, REPRO_ID, warmup=args.warmup, rep=args.rep
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
