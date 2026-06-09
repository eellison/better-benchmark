"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full sliced-vocabulary ignore-index cross-entropy mean by reading each logits row once with scalar online max and denominator accumulators, loading only the target logit, and reducing per-row losses plus valid counts to the final scalar; Inductor currently lowers the decomposed slice/view/amax/sub/exp/sum/log/log-softmax/gather/mask/sum/div graph as generic reductions and pointwise work that materializes and rereads a full log-softmax-sized intermediate. Inductor cannot do this today because its scheduler/template matching does not canonicalize log_softmax+gather+masked-mean into an online cross-entropy row template with a scalar reduction epilogue. The fix is NEW_PATTERN: add an Inductor lowering for this cross-entropy idiom that emits the online accumulator kernel and small final reduction directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

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
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _online_xent_rows_kernel(
    logits_ptr,
    labels_ptr,
    loss_ptr,
    valid_ptr,
    row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    label = tl.load(labels_ptr + row)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)

    row_start = row * row_stride
    x_label = tl.load(
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

    loss = row_max + tl.log(denom) - x_label
    loss = tl.where(is_valid, loss, 0.0)

    tl.store(loss_ptr + row, loss)
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


def oracle_online_cross_entropy_mean(
    addmm_1: torch.Tensor,
    arg7_1: torch.Tensor,
    block_n: int = 4096,
    num_warps: int = 8,
) -> torch.Tensor:
    assert addmm_1.is_cuda and arg7_1.is_cuda
    assert addmm_1.dtype == torch.float32 and addmm_1.ndim == 2
    assert arg7_1.dtype == torch.int64 and arg7_1.ndim == 2

    n_rows = addmm_1.shape[0]
    n_cols = addmm_1.shape[1] - 3
    assert arg7_1.numel() == n_rows

    loss_per_row = torch.empty((n_rows,), device=addmm_1.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=addmm_1.device, dtype=torch.float32)
    out = torch.empty((), device=addmm_1.device, dtype=torch.float32)

    _online_xent_rows_kernel[(n_rows,)](
        addmm_1,
        arg7_1,
        loss_per_row,
        valid_per_row,
        row_stride=addmm_1.stride(0),
        n_cols=n_cols,
        block_n=block_n,
        num_warps=num_warps,
    )

    block_m = triton.next_power_of_2(n_rows)
    _mean_reduce_kernel[(1,)](
        loss_per_row,
        valid_per_row,
        out,
        n_rows=n_rows,
        block_m=block_m,
        num_warps=8,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([8192, 50268], f32), T([8, 1024], i64), S([8, 1024, 50265]), S([8192, 50265]))")
def oracle_forward(inputs):
    addmm_1, arg7_1, _shape_param_0, _shape_param_1 = inputs
    return oracle_online_cross_entropy_mean(addmm_1, arg7_1)


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
