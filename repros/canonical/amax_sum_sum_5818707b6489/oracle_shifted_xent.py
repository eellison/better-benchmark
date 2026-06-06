"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete causal masked-language-model cross-entropy mean returned by Repro.forward, including the logits slice that drops the final two vocabulary columns, the labels-right-shift plus padded -100 ignore index, online row max/sum/log-prob accumulation, the ignored-row fill input, valid-count reduction, loss reduction, and final scalar division, whereas Inductor currently lowers the decomposed slice/view/pad/clone/amax/sub/exp/sum/log/gather/masked-sum/div graph as generic reductions and pointwise/gather kernels that materialize the full log-softmax; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize shifted ignore-index cross-entropy mean into one fused online row-reduction plus scalar epilogue; the fix is NEW_PATTERN: add an Inductor lowering for shifted log_softmax plus masked target gather plus mean loss/count that emits an online cross-entropy row kernel and small final reductions directly."""
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

BLOCK_N = 8192
NUM_WARPS_ROW = 16
BLOCK_M = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _shifted_xent_rows_kernel(
        logits_ptr,
        labels_ptr,
        fill_ptr,
        loss_ptr,
        valid_ptr,
        n_cols: tl.constexpr,
        logits_stride: tl.constexpr,
        seq_len: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // seq_len
        pos = row - batch * seq_len

        label = tl.load(
            labels_ptr + batch * seq_len + pos + 1,
            mask=pos < (seq_len - 1),
            other=-100,
        )
        is_valid = label != -100
        safe_label = tl.where(is_valid, label, 0)

        row_start = row * logits_stride
        target_logit = tl.load(
            logits_ptr + row_start + safe_label,
            mask=is_valid,
            other=0.0,
        ).to(tl.float32)

        row_max = tl.full([], -float("inf"), tl.float32)
        denom = tl.full([], 0.0, tl.float32)
        cols_base = tl.arange(0, block_n)

        for block_start in tl.range(0, n_cols, block_n):
            cols = block_start + cols_base
            mask = cols < n_cols
            x = tl.load(
                logits_ptr + row_start + cols,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)

            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(row_max, block_max)
            denom = denom * tl.exp(row_max - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
            row_max = new_max

        fill = tl.load(fill_ptr).to(tl.float32)
        loss = row_max + tl.log(denom) - target_logit
        tl.store(loss_ptr + row, tl.where(is_valid, loss, fill))
        tl.store(valid_ptr + row, tl.where(is_valid, 1.0, 0.0))

    @triton.jit
    def _partial_reduce_kernel(
        loss_ptr,
        valid_ptr,
        partial_loss_ptr,
        partial_valid_ptr,
        n_rows: tl.constexpr,
        block_m: tl.constexpr,
    ):
        block = tl.program_id(0)
        offsets = block * block_m + tl.arange(0, block_m)
        mask = offsets < n_rows
        losses = tl.load(loss_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        valid = tl.load(valid_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(partial_loss_ptr + block, tl.sum(losses, axis=0))
        tl.store(partial_valid_ptr + block, tl.sum(valid, axis=0))

    @triton.jit
    def _final_mean_kernel(
        partial_loss_ptr,
        partial_valid_ptr,
        out_ptr,
        n_blocks: tl.constexpr,
        block_b: tl.constexpr,
    ):
        offsets = tl.arange(0, block_b)
        mask = offsets < n_blocks
        losses = tl.load(partial_loss_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        valid = tl.load(partial_valid_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr, tl.sum(losses, axis=0) / tl.sum(valid, axis=0))


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def oracle_forward(inputs):
    """Run the full-scope oracle computation.

    SCOPE INVARIANT: accepts the exact input tuple from make_inputs() and
    returns the same scalar tensor as Repro.forward().
    """
    _require_triton_cuda()
    addmm_74, arg0_1, full_1, _shape_param_0, _shape_param_1 = inputs

    n_rows = int(_shape_param_1[0])
    n_cols = int(_shape_param_1[1])
    batch = int(_shape_param_0[0])
    seq_len = int(_shape_param_0[1])

    if addmm_74.ndim != 2 or addmm_74.dtype != torch.float32:
        raise ValueError(f"expected f32 2D logits input, got shape={tuple(addmm_74.shape)} dtype={addmm_74.dtype}")
    if arg0_1.shape != (batch, seq_len) or arg0_1.dtype != torch.int64:
        raise ValueError(f"expected int64 labels shape {(batch, seq_len)}, got {tuple(arg0_1.shape)} {arg0_1.dtype}")
    if full_1.shape != () or full_1.dtype != torch.float32:
        raise ValueError(f"expected scalar f32 fill input, got shape={tuple(full_1.shape)} dtype={full_1.dtype}")
    if n_rows != batch * seq_len or addmm_74.shape[0] != n_rows or addmm_74.shape[1] < n_cols:
        raise ValueError("shape parameters do not match logits/labels inputs")

    loss_per_row = torch.empty((n_rows,), device=addmm_74.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=addmm_74.device, dtype=torch.float32)
    n_blocks = triton.cdiv(n_rows, BLOCK_M)
    partial_loss = torch.empty((n_blocks,), device=addmm_74.device, dtype=torch.float32)
    partial_valid = torch.empty((n_blocks,), device=addmm_74.device, dtype=torch.float32)
    out = torch.empty((), device=addmm_74.device, dtype=torch.float32)

    _shifted_xent_rows_kernel[(n_rows,)](
        addmm_74,
        arg0_1,
        full_1,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
        logits_stride=addmm_74.stride(0),
        seq_len=seq_len,
        block_n=BLOCK_N,
        num_warps=NUM_WARPS_ROW,
    )
    _partial_reduce_kernel[(n_blocks,)](
        loss_per_row,
        valid_per_row,
        partial_loss,
        partial_valid,
        n_rows=n_rows,
        block_m=BLOCK_M,
        num_warps=4,
    )
    _final_mean_kernel[(1,)](
        partial_loss,
        partial_valid,
        out,
        n_blocks=n_blocks,
        block_b=triton.next_power_of_2(n_blocks),
        num_warps=1,
    )
    return out


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
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
