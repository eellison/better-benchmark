"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer masked-LM ignore-index cross-entropy mean returned by Repro.forward, including the flattened labels, logits slice that drops the final three columns, the two shape views, stable row logsumexp, safe ignored-label gather, valid-count reduction, loss reduction, and final scalar mean in Triton. Inductor currently lowers the decomposed view/slice/amax/sub/exp/sum/log/gather/mask/sum/count/div graph as generic reductions and pointwise work, materializing the full log-softmax-sized intermediate and reducing the valid count separately. Inductor cannot do this today because its pattern library does not canonicalize this decomposed ignore-index cross-entropy mean into an online row reduction that directly emits per-row losses plus the scalar count epilogue; the fix is NEW_PATTERN: add a guarded log_softmax plus masked-gather plus mean-cross-entropy lowering that preserves sliced-row strides and emits the online accumulator kernel directly."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _xent_rows_kernel(
        labels_ptr,
        logits_ptr,
        loss_ptr,
        valid_ptr,
        n_cols: tl.constexpr,
        logits_row_stride: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        label = tl.load(labels_ptr + row)
        is_valid = label != -100
        safe_label = tl.where(is_valid, label, 0)
        row_start = row * logits_row_stride

        target_logit = tl.load(
            logits_ptr + row_start + safe_label,
            mask=is_valid,
            other=0.0,
        ).to(tl.float32)

        row_max = tl.full([], -float("inf"), tl.float32)
        denom = tl.full([], 0.0, tl.float32)

        for block_start in tl.range(0, n_cols, block_n):
            cols = block_start + tl.arange(0, block_n)
            mask = cols < n_cols
            x = tl.load(
                logits_ptr + row_start + cols,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(row_max, block_max)
            denom = denom * tl.exp(row_max - new_max) + tl.sum(
                tl.exp(x - new_max),
                axis=0,
            )
            row_max = new_max

        loss = row_max + tl.log(denom) - target_logit
        tl.store(loss_ptr + row, tl.where(is_valid, loss, 0.0))
        tl.store(valid_ptr + row, tl.where(is_valid, 1.0, 0.0))

    @triton.jit
    def _mean_reduce_kernel(
        loss_ptr,
        valid_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        block_m: tl.constexpr,
    ):
        offsets = tl.arange(0, block_m)
        mask = offsets < n_rows
        losses = tl.load(loss_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        valid = tl.load(valid_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        total_loss = tl.sum(losses, axis=0)
        total_valid = tl.sum(valid, axis=0)
        tl.store(out_ptr, total_loss / total_valid)


def _launch_oracle(
    labels_1d: torch.Tensor,
    addmm: torch.Tensor,
    out: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    *,
    n_cols: int,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not labels_1d.is_cuda or not addmm.is_cuda:
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    if labels_1d.dtype != torch.int64 or labels_1d.ndim != 1:
        raise TypeError("expected flattened int64 labels")
    if addmm.dtype != torch.float32 or addmm.ndim != 2:
        raise TypeError("expected rank-2 float32 logits")
    if addmm.shape[1] < n_cols:
        raise ValueError(f"logits have {addmm.shape[1]} columns, need at least {n_cols}")

    n_rows = labels_1d.numel()
    if addmm.shape[0] != n_rows:
        raise ValueError(f"label/logit row mismatch: {n_rows} vs {addmm.shape[0]}")
    if out.shape != () or out.dtype != torch.float32:
        raise ValueError("output must be a scalar float32 tensor")
    if loss_per_row.shape != (n_rows,) or valid_per_row.shape != (n_rows,):
        raise ValueError("row scratch buffers must match the flattened label count")

    _xent_rows_kernel[(n_rows,)](
        labels_1d,
        addmm,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
        logits_row_stride=addmm.stride(0),
        block_n=block_n,
        num_warps=num_warps,
    )
    _mean_reduce_kernel[(1,)](
        loss_per_row,
        valid_per_row,
        out,
        n_rows=n_rows,
        block_m=triton.next_power_of_2(n_rows),
        num_warps=8,
    )
    return out


def oracle_online_cross_entropy_mean(
    arg7_1: torch.Tensor,
    addmm_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    *,
    block_n: int = 8192,
    num_warps: int = 16,
) -> torch.Tensor:
    labels_1d = arg7_1.view(-1)
    n_cols = int(_shape_param_1[1])
    n_rows = labels_1d.numel()
    out = torch.empty((), device=addmm_1.device, dtype=torch.float32)
    loss_per_row = torch.empty((n_rows,), device=addmm_1.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=addmm_1.device, dtype=torch.float32)
    return _launch_oracle(
        labels_1d,
        addmm_1,
        out,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
        block_n=block_n,
        num_warps=num_warps,
    )


@oracle_impl(hardware="H100", shapes="(T([8, 1024], i64), T([8192, 50268], f32), S([8, 1024, 50265]), S([8192, 50265]))")
def oracle_forward(inputs):
    """Run the exact scalar output computation from Repro()(*make_inputs())."""
    return oracle_online_cross_entropy_mean(*inputs)


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
