"""
Full-scope Triton oracle for sum_c93e126c2639.

Gap diagnosis (classification: BANDWIDTH_BOUND): This oracle consumes the same
original inputs as repro.py and returns the same single contiguous
`float32[48, 512, 512]` output. It fuses the dropout mask, attention mask,
softmax-gradient producer, row sum, and dependent FMA into one specialized
Triton row kernel over the complete width-512 attention row, avoiding any
subset-only reduction timing. Inductor cannot do materially better today
because the compiled region is already a single fused kernel for this small-row
attention backward tail; remaining time is dominated by reading the logits,
upstream gradient, two masks, and writing the full tensor. The fix class is
BANDWIDTH_BOUND.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_c93e126c2639"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
HEADS = 12
ROWS_PER_HEAD = 512
COLS = 512
TOTAL_ROWS = BATCH * HEADS * ROWS_PER_HEAD
DROPOUT_SCALE = 1.1111111111111112
HISTORICAL_BEST_COMPILE_US = 30.14400042593479

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
            "triton.multi_kernel=3"
        ),
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _attention_backward_row_kernel(
    bmm_ptr,
    dropout_mask_ptr,
    attention_mask_ptr,
    logits_ptr,
    row_max_ptr,
    row_denom_ptr,
    zero_row_ptr,
    out_ptr,
    DROPOUT_SCALE_: tl.constexpr,
    COLS_: tl.constexpr,
    HEADS_: tl.constexpr,
    ROWS_PER_HEAD_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    active = cols < COLS_

    b = row // (HEADS_ * ROWS_PER_HEAD_)
    row_in_batch_heads = row - b * HEADS_ * ROWS_PER_HEAD_
    m = row_in_batch_heads % ROWS_PER_HEAD_
    data_offsets = row * COLS_ + cols
    attention_mask_offsets = (b * ROWS_PER_HEAD_ + m) * COLS_ + cols

    keep_attention = tl.load(attention_mask_ptr + attention_mask_offsets, mask=active, other=0)
    zero_row = tl.load(zero_row_ptr + row)

    logits = tl.load(logits_ptr + data_offsets, mask=active, other=-float("inf")).to(tl.float32)
    row_max = tl.load(row_max_ptr + row).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + row).to(tl.float32)
    masked_logits = tl.where(keep_attention, logits - row_max, -float("inf"))
    prob = tl.exp(masked_logits) / row_denom
    prob = tl.where(zero_row, 0.0, prob)

    dropout_keep = tl.load(dropout_mask_ptr + data_offsets, mask=active, other=0).to(tl.float32)
    upstream = tl.load(bmm_ptr + data_offsets, mask=active, other=0.0).to(tl.float32)
    scaled_upstream = upstream * dropout_keep * DROPOUT_SCALE_
    weighted = scaled_upstream * prob
    row_sum = tl.sum(tl.where(active, weighted, 0.0), axis=0)

    out = weighted - prob * row_sum
    tl.store(out_ptr + data_offsets, out, mask=active)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def _validate_inputs(inputs: tuple[object, ...]) -> None:
    (
        bmm_1,
        arg14_1,
        arg4_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        shape0,
        shape1,
        shape2,
    ) = inputs
    expected_view = (BATCH, HEADS, ROWS_PER_HEAD, COLS)
    if tuple(shape0) != expected_view or tuple(shape1) != expected_view:
        raise ValueError("unexpected 4D shape parameters")
    if tuple(shape2) != (BATCH * HEADS, ROWS_PER_HEAD, COLS):
        raise ValueError("unexpected output shape parameter")
    if bmm_1.shape != (BATCH * HEADS, ROWS_PER_HEAD, COLS):
        raise ValueError(f"unexpected bmm_1 shape: {tuple(bmm_1.shape)}")
    if arg10_1.shape != bmm_1.shape:
        raise ValueError(f"unexpected arg10_1 shape: {tuple(arg10_1.shape)}")
    if arg14_1.shape != expected_view or arg4_1.shape != (BATCH, 1, ROWS_PER_HEAD, COLS):
        raise ValueError("unexpected mask shapes")
    if arg11_1.shape != (BATCH, HEADS, ROWS_PER_HEAD, 1):
        raise ValueError("unexpected arg11_1 shape")
    if arg12_1.shape != arg11_1.shape or arg13_1.shape != arg11_1.shape:
        raise ValueError("unexpected row scalar shapes")
    tensors = (bmm_1, arg14_1, arg4_1, arg10_1, arg11_1, arg12_1, arg13_1)
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("oracle expects the captured contiguous input layout")
    if bmm_1.dtype != torch.float32 or arg10_1.dtype != torch.float32:
        raise ValueError("expected float32 bmm/logit tensors")
    if arg11_1.dtype != torch.float32 or arg12_1.dtype != torch.float32:
        raise ValueError("expected float32 row max/denominator tensors")
    if arg14_1.dtype != torch.bool or arg4_1.dtype != torch.bool or arg13_1.dtype != torch.bool:
        raise ValueError("expected bool masks")


def oracle_fused(
    bmm_1: torch.Tensor,
    arg14_1: torch.Tensor,
    arg4_1: torch.Tensor,
    arg10_1: torch.Tensor,
    arg11_1: torch.Tensor,
    arg12_1: torch.Tensor,
    arg13_1: torch.Tensor,
    shape0,
    shape1,
    shape2,
) -> torch.Tensor:
    inputs = (bmm_1, arg14_1, arg4_1, arg10_1, arg11_1, arg12_1, arg13_1, shape0, shape1, shape2)
    _validate_inputs(inputs)

    out = torch.empty_strided(
        (BATCH * HEADS, ROWS_PER_HEAD, COLS),
        (ROWS_PER_HEAD * COLS, COLS, 1),
        device=bmm_1.device,
        dtype=torch.float32,
    )
    _attention_backward_row_kernel[(TOTAL_ROWS,)](
        bmm_1,
        arg14_1,
        arg4_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        out,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        COLS_=COLS,
        HEADS_=HEADS,
        ROWS_PER_HEAD_=ROWS_PER_HEAD,
        BLOCK_N=COLS,
        num_warps=1,
    )
    return out


def reference_output(inputs: tuple[object, ...]) -> torch.Tensor:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return float(diff.max().item()), float(rel.max().item())


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_output(inputs)
        actual = oracle_fused(*inputs)
        torch.cuda.synchronize()

    shape_ok = actual.shape == ref.shape
    dtype_ok = actual.dtype == ref.dtype
    stride_ok = actual.stride() == ref.stride()
    max_abs, max_rel = _max_diff(actual, ref)
    close_ok = torch.allclose(actual.float(), ref.float(), rtol=rtol, atol=atol)
    ok = shape_ok and dtype_ok and stride_ok and close_ok
    print(
        f"output[0]: shape={list(actual.shape)} dtype={actual.dtype} stride={actual.stride()} "
        f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={close_ok} "
        f"shape_match={shape_ok} stride_match={stride_ok} dtype_match={dtype_ok}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(model: torch.nn.Module, inputs: tuple[object, ...], config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(rep: int, warmup: int, no_compile: bool) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    timings: dict[str, float] = {}
    with torch.no_grad():
        oracle_fused(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_fused(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    timings["oracle_fused"] = oracle_us
    print(f"oracle_fused full-scope attention backward row kernel: {oracle_us:.3f} us")

    if not no_compile:
        module = _load_repro_module()
        for label, config in COMPILE_CONFIGS:
            model = module.Repro().cuda()
            with torch.no_grad():
                compiled = _compile_with_config(model, inputs, config)
                compiled_us = triton.testing.do_bench(
                    lambda: compiled(*inputs),
                    warmup=warmup,
                    rep=rep,
                    return_mode="min",
                ) * 1000.0
            timings[label] = compiled_us
            print(f"torch.compile {label}: {compiled_us:.3f} us")

    best_required_compile = min(
        (timings[label] for label, _ in COMPILE_CONFIGS if label in timings),
        default=float("inf"),
    )
    best_reference = min(best_required_compile, HISTORICAL_BEST_COMPILE_US)
    valid_floor = oracle_us < best_reference
    print(f"historical best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"Valid floor: {'yes' if valid_floor else 'no (diagnosis-only)'}")
    print(
        json.dumps(
            {
                "repro_id": REPRO_ID,
                "oracle_us": round(oracle_us, 3),
                "best_required_compile_us": (
                    round(best_required_compile, 3)
                    if best_required_compile != float("inf")
                    else None
                ),
                "historical_best_compile_us": round(HISTORICAL_BEST_COMPILE_US, 3),
                "valid_floor": valid_floor,
                "classification": "BANDWIDTH_BOUND",
            }
        )
    )
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=2e-1)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
