"""Gap diagnosis (classification: NEW_PATTERN): this full-scope oracle computes `Repro.forward` in one Triton kernel by reducing `arg0_1 != 0` to the last nonzero token per batch, gathering the corresponding two logits through the dynamic `iota_1` row index, evaluating the ignore-index two-class log-softmax/NLL mean, and writing the `[32,2]` eq side output, while Inductor currently lowers the argmax/gather/log_softmax/NLL/count/eq graph as generic reductions and pointwise work instead of recognizing the whole last-token sequence-classification cross-entropy tail; the fix is to add a guarded lowering for this argmax-selected two-class cross-entropy plus side-mask pattern that emits a fused reduction/epilogue kernel preserving the ignore-index and side-output semantics."""
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


REPRO_ID = "amax_argmax_sum_5dfabd070411"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
SEQ_LEN = 128
N_CLASSES = 2


if triton is not None:

    @triton.jit
    def _last_token_xent_eq_kernel(
        mm_ptr,
        token_ptr,
        row_index_ptr,
        label_ptr,
        full_ptr,
        loss_out_ptr,
        eq_out_ptr,
        BLOCK_B: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batches = tl.arange(0, BLOCK_B)
        seq = tl.arange(0, BLOCK_S)
        token_offsets = batches[:, None] * BLOCK_S + seq[None, :]
        tokens = tl.load(token_ptr + token_offsets)
        positions = tl.where(tokens != 0, seq[None, :], 0)
        last_pos = tl.max(positions, axis=1)

        row_index = tl.load(row_index_ptr + batches)
        label = tl.load(label_ptr + batches)
        valid = label != -100
        safe_label = tl.where(valid, label, 0)

        classes = tl.arange(0, BLOCK_C)
        logits = tl.load(
            mm_ptr
            + row_index[:, None] * (BLOCK_S * BLOCK_C)
            + last_pos[:, None] * BLOCK_C
            + classes[None, :]
        ).to(tl.float32)

        row_max = tl.max(logits, axis=1)
        denom = tl.sum(tl.exp(logits - row_max[:, None]), axis=1)
        target = tl.load(
            mm_ptr + row_index * (BLOCK_S * BLOCK_C) + last_pos * BLOCK_C + safe_label
        ).to(tl.float32)
        losses = row_max + tl.log(denom) - target
        ignored_value = tl.load(full_ptr).to(tl.float32)
        losses = tl.where(valid, losses, ignored_value)

        total_loss = tl.sum(losses, axis=0)
        total_valid = tl.sum(tl.where(valid, 1.0, 0.0), axis=0)
        tl.store(loss_out_ptr, total_loss / total_valid)

        eq_values = safe_label[:, None] == classes[None, :]
        tl.store(eq_out_ptr + batches[:, None] * BLOCK_C + classes[None, :], eq_values)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _validate_shape_param(name: str, value, expected: tuple[int, ...]) -> None:
    if tuple(int(dim) for dim in value) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(value)}")


def _validate_inputs(inputs) -> None:
    mm_72, arg0_1, iota_1, arg342_1, full_1, shape0, shape1, shape2 = inputs
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not mm_72.is_cuda:
        raise RuntimeError("CUDA inputs are required for this Triton oracle")
    if mm_72.shape != (BATCH * SEQ_LEN, N_CLASSES) or mm_72.dtype != torch.float32:
        raise ValueError(f"expected mm_72 f32[{BATCH * SEQ_LEN}, {N_CLASSES}]")
    if arg0_1.shape != (BATCH, SEQ_LEN) or arg0_1.dtype != torch.int64:
        raise ValueError(f"expected arg0_1 i64[{BATCH}, {SEQ_LEN}]")
    if iota_1.shape != (BATCH,) or iota_1.dtype != torch.int64:
        raise ValueError(f"expected iota_1 i64[{BATCH}]")
    if arg342_1.shape != (BATCH,) or arg342_1.dtype != torch.int64:
        raise ValueError(f"expected arg342_1 i64[{BATCH}]")
    if full_1.shape != () or full_1.dtype != torch.float32:
        raise ValueError("expected full_1 scalar f32")
    _validate_shape_param("_shape_param_0", shape0, (BATCH, SEQ_LEN, N_CLASSES))
    _validate_shape_param("_shape_param_1", shape1, (1, N_CLASSES))
    _validate_shape_param("_shape_param_2", shape2, (BATCH, N_CLASSES))


@oracle_impl(hardware="H100", shapes="(T([4096, 2], f32), T([32, 128], i64), T([32], i64, gen=Index(32)), T([32], i64, gen=Index(2)), T([], f32), S([32, 128, 2]), S([1, 2]), S([32, 2]))")
def oracle_forward(inputs):
    """Run the full argmax-gather, two-class cross-entropy mean, and eq side output."""
    _validate_inputs(inputs)
    mm_72, arg0_1, iota_1, arg342_1, full_1, *_ = inputs
    loss = torch.empty((), device=mm_72.device, dtype=torch.float32)
    eq = torch.empty((BATCH, N_CLASSES), device=mm_72.device, dtype=torch.bool)
    _last_token_xent_eq_kernel[(1,)](
        mm_72,
        arg0_1,
        iota_1,
        arg342_1,
        full_1,
        loss,
        eq,
        BLOCK_B=BATCH,
        BLOCK_S=SEQ_LEN,
        BLOCK_C=N_CLASSES,
        num_warps=8,
    )
    return loss, eq


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=200, help="Benchmark repetitions")
    parser.add_argument("--no-skip-stochastic", action="store_true")
    parser.add_argument("--all-shapes", action="store_true")
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
