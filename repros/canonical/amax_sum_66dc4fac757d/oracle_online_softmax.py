"""
Gap diagnosis (classification: NEW_PATTERN): this oracle covers the full
Reformer logsumexp/softmax-style materialization from repro.py, including the
fp16 `[768,64,128]` input view as `[1,12,64,64,128]`, the fp32 max/exp/sum/log
over the last dimension, the `abs(max) == inf` replacement with zero, the fp16
logsumexp cast, the second fp16 subtract/exp, and the final contiguous
`[768,64,128]` output view. It differs from Inductor only by using a dedicated
multi-row Triton template that keeps each 128-wide row in registers
across the reduction and epilogue instead of relying on the generic generated
online-softmax schedule. Inductor cannot do this today because it does not
canonicalize this exact max/sum/log, fp16 logsumexp cast, and second fp16
exp epilogue into a multi-row persistent softmax template; the existing generic
single-kernel reduction schedule covers the scope but leaves small-row overhead
on the table. The fix class is NEW_PATTERN: add a specialized small-row
logsumexp-to-exp lowering that preserves the fp16 rounding points and final
view while processing multiple rows per Triton program.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_66dc4fac757d"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 1 * 12 * 64 * 64
COLS = 128
INPUT_SHAPE = (768, 64, 128)
VIEW_SHAPE = (1, 12, 64, 64, 128)
OUT_SHAPE = INPUT_SHAPE
OUT_STRIDE = (64 * 128, 128, 1)

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
        "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
        "triton.multi_kernel=3",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


@triton.jit
def _logsumexp_exp_rows_kernel(
    x_ptr,
    out_ptr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    cols = tl.arange(0, block_cols)
    mask = (row_offsets[:, None] < 49152) & (cols[None, :] < 128)

    i0 = row_offsets // 64
    i1 = row_offsets - i0 * 64

    x_offsets = i0[:, None] * x_s0 + i1[:, None] * x_s1 + cols[None, :] * x_s2
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=-float("inf"))
    x_f32 = x_vals.to(tl.float32)
    x_f32 = tl.where(mask, x_f32, -float("inf"))

    row_max = tl.max(x_f32, axis=1)
    stable_max = tl.where(tl.abs(row_max) == float("inf"), 0.0, row_max)
    exp_shifted = tl.exp(x_f32 - stable_max[:, None])
    denom = tl.sum(exp_shifted, axis=1)
    logsumexp = tl.log(denom) + stable_max

    # repro.py casts logsumexp to fp16, then performs the final subtract/exp
    # on fp16 tensors. Round those two intermediate values the same way.
    logsumexp_h = logsumexp.to(tl.float16)
    shifted_h = (x_f32 - logsumexp_h[:, None].to(tl.float32)).to(tl.float16)
    out_vals = tl.exp(shifted_h.to(tl.float32))

    out_offsets = i0[:, None] * out_s0 + i1[:, None] * out_s1 + cols[None, :] * out_s2
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_inputs(module, seed: int) -> tuple:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = module.make_inputs()
    return tuple(
        x.cuda() if isinstance(x, torch.Tensor) and not x.is_cuda else x
        for x in inputs
    )


def _validate_shape_param(name: str, actual, expected: tuple[int, ...]) -> None:
    if actual is None:
        return
    got = tuple(int(dim) for dim in actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(
    bmm_10: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
) -> None:
    if not bmm_10.is_cuda:
        raise RuntimeError("CUDA tensor input is required")
    if bmm_10.dtype != torch.float16:
        raise TypeError(f"expected bmm_10 fp16, got {bmm_10.dtype}")
    if tuple(bmm_10.shape) != INPUT_SHAPE:
        raise ValueError(f"expected bmm_10 shape {INPUT_SHAPE}, got {tuple(bmm_10.shape)}")
    if bmm_10.stride() != OUT_STRIDE:
        raise ValueError(f"expected contiguous input stride {OUT_STRIDE}, got {bmm_10.stride()}")
    _validate_shape_param("_shape_param_0", _shape_param_0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, VIEW_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


def _launch_oracle(
    bmm_10: torch.Tensor,
    out: torch.Tensor,
    *,
    block_rows: int,
    block_cols: int,
    num_warps: int,
) -> torch.Tensor:
    if block_rows <= 0 or block_rows & (block_rows - 1):
        raise ValueError(f"block_rows must be a positive power of two, got {block_rows}")
    if block_cols < COLS or block_cols & (block_cols - 1):
        raise ValueError(f"block_cols must be a power of two >= {COLS}, got {block_cols}")
    if out.shape != OUT_SHAPE or out.dtype != torch.float16 or not out.is_cuda:
        raise ValueError("preallocated output must be CUDA fp16 with repro output shape")
    if out.stride() != OUT_STRIDE:
        raise ValueError(f"preallocated output stride mismatch: expected {OUT_STRIDE}, got {out.stride()}")

    grid = (triton.cdiv(ROWS, block_rows),)
    _logsumexp_exp_rows_kernel[grid](
        bmm_10,
        out,
        x_s0=bmm_10.stride(0),
        x_s1=bmm_10.stride(1),
        x_s2=bmm_10.stride(2),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        block_rows=block_rows,
        block_cols=block_cols,
        num_warps=num_warps,
    )
    return out


def oracle_online_softmax(
    bmm_10: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
    *,
    block_rows: int = 8,
    block_cols: int | None = None,
    num_warps: int = 4,
) -> torch.Tensor:
    _validate_inputs(bmm_10, _shape_param_0, _shape_param_1, _shape_param_2)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_10.device,
        dtype=torch.float16,
    )
    actual_block_cols = block_cols if block_cols is not None else triton.next_power_of_2(COLS)
    return _launch_oracle(
        bmm_10,
        out,
        block_rows=block_rows,
        block_cols=actual_block_cols,
        num_warps=num_warps,
    )


def _bench_cuda_graph(fn, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _compile_with_config(module, inputs: tuple, config: dict[str, object], warmup: int):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = module.Repro().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    max_abs = torch.nan_to_num(diff, nan=0.0).max().item()
    rel = diff / expected_f32.abs().clamp_min(1e-8)
    max_rel = torch.nan_to_num(rel, nan=0.0).max().item()
    return max_abs, max_rel


def run_check(block_rows: int, block_cols: int | None, num_warps: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=1234)
    _validate_inputs(*inputs)
    model = module.Repro().cuda()

    with torch.no_grad():
        ref = model(*inputs)
        got = oracle_online_softmax(
            *inputs,
            block_rows=block_rows,
            block_cols=block_cols,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()

    max_abs, max_rel = _max_diff(got, ref)
    shape_match = tuple(got.shape) == tuple(ref.shape)
    dtype_match = got.dtype == ref.dtype
    stride_match = got.stride() == ref.stride()
    value_match = torch.allclose(got.float(), ref.float(), rtol=5e-3, atol=5e-4)
    ok = shape_match and dtype_match and stride_match and value_match

    print(
        "check full-scope Reformer logsumexp softmax: "
        f"shape={tuple(got.shape)} dtype={got.dtype} stride={got.stride()} "
        f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
        f"shape_match={shape_match} dtype_match={dtype_match} "
        f"stride_match={stride_match} allclose={value_match}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def run_bench(
    block_cols: int | None,
    block_rows: int,
    num_warps: int,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321)
    _validate_inputs(*inputs)
    bmm_10 = inputs[0]
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_10.device,
        dtype=torch.float16,
    )
    actual_block_cols = block_cols if block_cols is not None else triton.next_power_of_2(COLS)
    traffic_bytes = bmm_10.numel() * 2 * 2

    print(
        "oracle shape: "
        f"bmm_10=f16{tuple(bmm_10.shape)} stride={bmm_10.stride()} "
        f"view=f16{VIEW_SHAPE} out=f16{OUT_SHAPE} stride={out.stride()}"
    )
    print(f"single-kernel logical traffic: {traffic_bytes / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(
                bmm_10,
                out,
                block_rows=block_rows,
                block_cols=actual_block_cols,
                num_warps=num_warps,
            ),
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = traffic_bytes / (oracle_us * 1e-6) / 1e12
    print(
        "oracle full-scope logsumexp softmax: "
        f"{oracle_us:.3f} us ({oracle_bw:.3f} TB/s logical bytes)"
    )
    print(f"oracle_us={oracle_us:.3f}")

    if no_compile:
        return

    print("CUDA graph replay timings cover the same repro.py logsumexp, fp16 cast, second exp, and view")
    compile_times: list[tuple[str, float]] = []
    for label, config in COMPILE_CONFIGS:
        try:
            compiled = _compile_with_config(module, inputs, config, warmup)
            us = _bench_cuda_graph(lambda: compiled(*inputs), warmup=warmup, rep=rep)
            compile_times.append((label, us))
            print(f"torch.compile {label}: {us:.3f} us")
        except Exception as exc:
            print(f"torch.compile {label}: FAILED ({exc})")

    if compile_times:
        best_compile = min(us for _, us in compile_times)
        valid_floor = oracle_us < min(us for _, us in compile_times)
        print(f"best_required_compile_us={best_compile:.3f}")
        print(f"valid_floor={valid_floor}")
        if not valid_floor:
            print("diagnosis_only: oracle is not a true floor because a required compile config is faster")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="benchmark repetitions")
    parser.add_argument("--block-rows", type=int, default=8, help="Triton rows per program")
    parser.add_argument("--block-cols", type=int, default=None, help="Triton column tile size")
    parser.add_argument("--num-warps", type=int, default=4, help="Triton warps per program")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile baselines for the requested configs",
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check:
        ok = run_check(
            block_rows=args.block_rows,
            block_cols=args.block_cols,
            num_warps=args.num_warps,
        )
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench(
            block_cols=args.block_cols,
            block_rows=args.block_rows,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
