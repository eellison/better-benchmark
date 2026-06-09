"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT masked-LM captured scope from repro.py, including the logits slice, bias add, returned biased logits view, ignore-index cross-entropy mean, and returned materialized softmax probabilities, whereas Inductor currently lowers the same slice/add/log_softmax/gather/masked-mean/probability-materialization graph as generic reductions and pointwise/layout work with separate materialized intermediates; Inductor cannot do this today because its pattern library does not recognize biased log_softmax feeding both gather/masked mean and a sibling exp output as one multi-output online-softmax row template; the fix is NEW_PATTERN: add a guarded biased softmax-cross-entropy lowering that shares row max/sum accumulators across the loss path and required materialized probability/logit outputs."""
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
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "NEW_PATTERN"
ACTIONABLE = True
BLOCK_N = 2048
NUM_WARPS = 8


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _full_online_softmax_xent_kernel(
        mm_ptr,
        bias_ptr,
        labels_ptr,
        fill_ptr,
        logits_out_ptr,
        probs_out_ptr,
        loss_ptr,
        valid_ptr,
        n_cols: tl.constexpr,
        mm_row_stride: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols_base = tl.arange(0, block_n)
        mm_row_start = row * mm_row_stride
        out_row_start = row * n_cols

        label = tl.load(labels_ptr + row)
        is_valid = label != -100
        safe_label = tl.where(is_valid, label, 0)
        fill = tl.load(fill_ptr).to(tl.float32)
        target = tl.load(
            mm_ptr + mm_row_start + safe_label,
            mask=is_valid,
            other=0.0,
        ).to(tl.float32)
        target += tl.load(bias_ptr + safe_label, mask=is_valid, other=0.0).to(tl.float32)

        row_max = tl.full([], -float("inf"), tl.float32)
        denom = tl.full([], 0.0, tl.float32)

        for block_start in tl.range(0, n_cols, block_n):
            cols = block_start + cols_base
            mask = cols < n_cols
            x = tl.load(
                mm_ptr + mm_row_start + cols,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            x = x + b
            tl.store(logits_out_ptr + out_row_start + cols, x, mask=mask)

            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(row_max, block_max)
            denom = denom * tl.exp(row_max - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
            row_max = new_max

        loss = row_max + tl.log(denom) - target
        loss = tl.where(is_valid, loss, fill)
        tl.store(loss_ptr + row, loss)
        tl.store(valid_ptr + row, tl.where(is_valid, 1.0, 0.0))

        for block_start in tl.range(0, n_cols, block_n):
            cols = block_start + cols_base
            mask = cols < n_cols
            x = tl.load(
                logits_out_ptr + out_row_start + cols,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            probs = tl.exp(x - row_max) / denom
            tl.store(probs_out_ptr + out_row_start + cols, probs, mask=mask)

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


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _launch_oracle(
    mm: torch.Tensor,
    bias: torch.Tensor,
    labels: torch.Tensor,
    fill: torch.Tensor,
    logits_out: torch.Tensor,
    probs_out: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    mean_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton_cuda()
    n_rows = labels.numel()
    n_cols = bias.numel()

    if mm.ndim != 2 or mm.dtype != torch.float32:
        raise ValueError(f"expected f32 2D mm, got shape={tuple(mm.shape)} dtype={mm.dtype}")
    if bias.ndim != 1 or bias.dtype != torch.float32:
        raise ValueError(f"expected f32 1D bias, got shape={tuple(bias.shape)} dtype={bias.dtype}")
    if labels.dtype != torch.int64:
        raise ValueError(f"expected int64 labels, got dtype={labels.dtype}")
    if fill.shape != () or fill.dtype != torch.float32:
        raise ValueError(f"expected scalar f32 fill, got shape={tuple(fill.shape)} dtype={fill.dtype}")
    if mm.shape[0] != n_rows or mm.shape[1] < n_cols:
        raise ValueError(f"mm shape {tuple(mm.shape)} does not cover labels={n_rows}, bias={n_cols}")
    if logits_out.numel() != n_rows * n_cols or probs_out.shape != (n_rows, n_cols):
        raise ValueError("output shapes do not match the logical repro outputs")

    _full_online_softmax_xent_kernel[(n_rows,)](
        mm,
        bias,
        labels,
        fill,
        logits_out,
        probs_out,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
        mm_row_stride=mm.stride(0),
        block_n=BLOCK_N,
        num_warps=NUM_WARPS,
    )
    _mean_reduce_kernel[(1,)](
        loss_per_row,
        valid_per_row,
        mean_out,
        n_rows=n_rows,
        block_m=triton.next_power_of_2(n_rows),
        num_warps=8,
    )
    return logits_out, mean_out, probs_out


@oracle_impl(hardware="H100", shapes="(T([32768, 30524], f32), T([30522], f32), T([256, 128], i64), T([], f32), S([256, 128, 30522]), S([32768, 30522]), S([256, 128, 30522]))")
def oracle_forward(inputs, *, block_n: int = 2048, num_warps: int = 8):
    """Run the full-scope oracle computation.

    SCOPE INVARIANT: this accepts the exact input tuple from make_inputs() and
    returns the same three tensors as Repro.forward().
    """
    mm, bias, labels, fill, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    n_rows = labels.numel()
    n_cols = bias.numel()
    logits_out = torch.empty(tuple(_shape_param_2), device=mm.device, dtype=torch.float32)
    probs_out = torch.empty((n_rows, n_cols), device=mm.device, dtype=torch.float32)
    loss_per_row = torch.empty((n_rows,), device=mm.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=mm.device, dtype=torch.float32)
    mean_out = torch.empty((), device=mm.device, dtype=torch.float32)

    labels_1d = labels.reshape(-1)
    del block_n, num_warps
    return _launch_oracle(
        mm,
        bias,
        labels_1d,
        fill,
        logits_out,
        probs_out,
        loss_per_row,
        valid_per_row,
        mean_out,
    )


def _normalize_outputs(out) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        flat: list[torch.Tensor] = []
        for item in out:
            flat.extend(_normalize_outputs(item))
        return flat
    return []


def _floating_close_stats(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    chunk_elements: int = 8 * 1024 * 1024,
) -> tuple[float, bool]:
    expected_flat = expected.reshape(-1)
    actual_flat = actual.reshape(-1)
    max_diff = 0.0
    values_ok = True

    for start in range(0, expected_flat.numel(), chunk_elements):
        end = min(start + chunk_elements, expected_flat.numel())
        expected_chunk = expected_flat[start:end].to(device="cpu", dtype=torch.float32)
        actual_chunk = actual_flat[start:end].to(device="cpu", dtype=torch.float32)
        diff = (expected_chunk - actual_chunk).abs()
        if diff.numel() > 0:
            max_diff = max(max_diff, diff.max().item())
        if values_ok:
            tolerance = atol + rtol * expected_chunk.abs()
            values_ok = not bool((diff > tolerance).any().item())

    return max_diff, values_ok


def _run_check(
    instance: torch.nn.Module,
    inputs,
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        torch.cuda.synchronize()
        expected_outputs = [out.detach().cpu() for out in _normalize_outputs(expected)]
        del expected
        torch.cuda.empty_cache()

        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    actual_outputs = _normalize_outputs(actual)
    if len(expected_outputs) != len(actual_outputs):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_outputs)} outputs, "
            f"eager produces {len(expected_outputs)}"
        )
        return False

    all_ok = True
    for i, (expected_i, actual_i) in enumerate(zip(expected_outputs, actual_outputs)):
        shape_ok = expected_i.shape == actual_i.shape
        dtype_ok = expected_i.dtype == actual_i.dtype
        layout_ok = expected_i.stride() == actual_i.stride()

        if not shape_ok:
            print(
                f"  output {i} values: FAIL "
                f"(expected_shape={list(expected_i.shape)} oracle_shape={list(actual_i.shape)})"
            )
            all_ok = False
            continue

        if expected_i.is_floating_point():
            max_diff, values_ok = _floating_close_stats(
                expected_i,
                actual_i,
                atol=atol,
                rtol=rtol,
            )
        else:
            max_diff = 0.0
            values_ok = torch.equal(expected_i.cpu(), actual_i.cpu())

        value_status = "PASS" if values_ok and dtype_ok else "FAIL"
        layout_status = "PASS" if layout_ok else "FAIL"
        print(
            f"  output {i} values: {value_status} "
            f"(shape={list(expected_i.shape)} dtype={expected_i.dtype} "
            f"oracle_dtype={actual_i.dtype} max_diff={max_diff:.2e})"
        )
        print(
            f"  output {i} layout: {layout_status} "
            f"(expected_stride={expected_i.stride()}, oracle_stride={actual_i.stride()})"
        )
        all_ok = all_ok and values_ok and dtype_ok and layout_ok

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle via oracle_harness")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
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

    _require_triton_cuda()
    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    with torch.no_grad():
        if args.check:
            print(f"Checking {REPRO_ID}...")
            ok = _run_check(instance, inputs, atol=args.atol, rtol=args.rtol)
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
                true_floor = result["status"] == "GOOD"
                print(f"classification: {CLASSIFICATION}")
                print(f"true_floor: {'yes' if true_floor else 'no'} ({result['status']})")
                print(f"actionable: {'yes' if ACTIONABLE and true_floor else 'no'}")
                if not true_floor:
                    print(
                        "diagnosis_only: not_true_floor because compiled Inductor "
                        "is at least as fast as this full-scope oracle"
                    )


if __name__ == "__main__":
    main()
