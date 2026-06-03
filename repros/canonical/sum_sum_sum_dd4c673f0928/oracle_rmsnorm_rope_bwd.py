"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Qwen grouped-RoPE plus two-branch RMSNorm-backward return tuple by rebuilding the rotary half-rotation algebraically, reusing each row producer for the hidden-dimension RMSNorm dot product, the transposed input-gradient side output, and the sibling `[128]` weight-gradient reductions, whereas Inductor currently materializes the grouped-head sum, rotary `slice_scatter` reconstructions, permutes, RMSNorm row gradients, and sibling sums as separate generic pointwise/reduction/layout kernels; Inductor cannot do this today because the scheduler/codegen cannot represent a multi-branch row-persistent RMSNorm-backward fusion whose rotary producer has a grouped-head reduction, materialized layout-changing side outputs, and cross-row weight-gradient reducers; the fix is SCHEDULER_FUSION: add a fused grouped-RoPE/RMSNorm-backward template that keeps each row dot product in registers, stores the transposed gradients directly, and emits coordinated partial reductions for the two weight gradients."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch


REPRO_ID = "sum_sum_sum_dd4c673f0928"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_qwen_qwen3-0.6b_001_6a9f0787"

BATCH = 4
SEQ = 512
TOKENS = BATCH * SEQ
QUERY_HEADS = 16
KEY_VALUE_HEADS = 8
HEADS_PER_GROUP = QUERY_HEADS // KEY_VALUE_HEADS
HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
QUERY_HIDDEN = QUERY_HEADS * HEAD_DIM
KEY_VALUE_HIDDEN = KEY_VALUE_HEADS * HEAD_DIM

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        config = next(iter(configs.values()))
        config = {
            "inputs": [
                {**spec, "device": str(device)}
                if isinstance(spec, dict) and spec.get("kind") == "tensor"
                else spec
                for spec in config["inputs"]
            ]
        }
        inputs = make_inputs_from_config(config)
    else:
        inputs = _load_repro_module().make_inputs()

    return tuple(
        value.to(device=device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    )


def _rotary_tables(arg2_1: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(SEQ, device=arg2_1.device, dtype=torch.float32)
    freqs = positions[:, None] * arg2_1[None, :]
    angles = torch.cat((freqs, freqs), dim=1)
    sin = torch.sin(angles).to(dtype)
    cos = torch.cos(angles).to(dtype)
    return sin[None, None, :, :], cos[None, None, :, :]


def _apply_rope_backward(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    x_sin = x * sin
    rotated = torch.cat((x_sin[..., HALF_DIM:], -x_sin[..., :HALF_DIM]), dim=-1)
    return rotated + x * cos


def _rmsnorm_backward_branch(
    row_grad: torch.Tensor,
    weight: torch.Tensor,
    saved_x: torch.Tensor,
    inv_rstd: torch.Tensor,
    heads: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = saved_x.view(BATCH, SEQ, heads, HEAD_DIM)
    x_f32 = x.float()

    xhat = (x_f32 * inv_rstd).to(torch.bfloat16)
    weight_grad = (row_grad * xhat).sum(dim=(0, 1, 2), keepdim=True).view(HEAD_DIM)

    dy_weight = (row_grad * weight).float()
    dot = (dy_weight * x_f32).sum(dim=3, keepdim=True)
    correction_scale = (((dot * -0.5) * (inv_rstd**3)).expand_as(dy_weight)) / HEAD_DIM
    correction = correction_scale * ((x_f32**1.0) * 2.0)
    input_grad = dy_weight * inv_rstd + correction
    input_grad_t = (
        input_grad.to(torch.bfloat16)
        .contiguous()
        .view(BATCH, SEQ, hidden)
        .view(TOKENS, hidden)
        .permute(1, 0)
    )
    return weight_grad, input_grad_t


def oracle_rmsnorm_rope_bwd(
    getitem_1: torch.Tensor,
    arg2_1: torch.Tensor,
    getitem: torch.Tensor,
    arg304_1: torch.Tensor,
    arg858_1: torch.Tensor,
    arg859_1: torch.Tensor,
    arg302_1: torch.Tensor,
    arg856_1: torch.Tensor,
    arg857_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
    _shape_param_8,
    _shape_param_9,
    _shape_param_10,
    _shape_param_11,
    _shape_param_12,
    _shape_param_13,
    _shape_param_14,
    _shape_param_15,
    _shape_param_16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
        _shape_param_9,
        _shape_param_10,
        _shape_param_11,
        _shape_param_12,
        _shape_param_13,
        _shape_param_14,
        _shape_param_15,
        _shape_param_16,
    )

    sin, cos = _rotary_tables(arg2_1, getitem.dtype)
    grouped = (
        getitem_1.view(BATCH, KEY_VALUE_HEADS, HEADS_PER_GROUP, SEQ, HEAD_DIM)
        .sum(dim=2, keepdim=True)
        .squeeze(2)
    )

    kv_row_grad = _apply_rope_backward(grouped, sin, cos).permute(0, 2, 1, 3)
    q_row_grad = _apply_rope_backward(getitem, sin, cos).permute(0, 2, 1, 3)

    kv_weight_grad, kv_input_grad_t = _rmsnorm_backward_branch(
        kv_row_grad,
        arg304_1,
        arg858_1,
        arg859_1,
        KEY_VALUE_HEADS,
        KEY_VALUE_HIDDEN,
    )
    q_weight_grad, q_input_grad_t = _rmsnorm_backward_branch(
        q_row_grad,
        arg302_1,
        arg856_1,
        arg857_1,
        QUERY_HEADS,
        QUERY_HIDDEN,
    )
    return kv_weight_grad, kv_input_grad_t, q_weight_grad, q_input_grad_t


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device=device)
    out = model(*inputs)
    if not isinstance(out, tuple):
        raise TypeError(f"expected tuple output from repro, got {type(out)!r}")
    return out


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_rmsnorm_rope_bwd(*inputs)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == ref.stride()
        dtype_ok = got.dtype == ref.dtype
        shape_ok = got.shape == ref.shape
        ok = ok and value_ok and stride_ok and dtype_ok and shape_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"expected_stride={ref.stride()} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={value_ok} "
            f"shape_match={shape_ok} stride_match={stride_ok} dtype_match={dtype_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def run_bench(device: torch.device, warmup: int, rep: int, include_repro: bool) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        oracle_fn = lambda: oracle_rmsnorm_rope_bwd(*inputs)
        oracle_fn()
        synchronize(device)
        oracle_us = benchmark(oracle_fn, device, warmup, rep)

        print(
            f"oracle_rmsnorm_rope_bwd: {oracle_us:.3f} us "
            f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
        )

        if include_repro:
            module = _load_repro_module()
            module.device = lambda *unused_args, **unused_kwargs: device
            repro = module.Repro().to(device=device)
            repro_fn = lambda: repro(*inputs)
            repro_fn()
            synchronize(device)
            repro_us = benchmark(repro_fn, device, warmup, rep)
            print(
                f"repro_eager: {repro_us:.3f} us "
                f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare full Repro.forward outputs against the oracle")
    parser.add_argument("--bench", action="store_true", help="benchmark the structured oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--include-repro", action="store_true", help="also benchmark eager repro.py")
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, warmup=args.warmup, rep=args.rep, include_repro=args.include_repro)


if __name__ == "__main__":
    main()
