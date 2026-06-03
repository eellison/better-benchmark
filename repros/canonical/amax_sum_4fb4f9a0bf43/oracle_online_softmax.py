"""
Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle covers the full T5 additive-score softmax materialization from repro.py, including the fp16 `[H,2048,2048] + [1,H,2048,2048]` add rounded through fp16, fp32 last-dimension softmax, fp16 cast, and contiguous `[H,2048,2048]` output view, using one Triton row program per query row instead of Inductor's generated softmax schedule. Inductor can already keep this default shape in a single compiled kernel under the required combo-looped configuration, so the remaining difference is a hand-written row template rather than a missing fusion across surrounding ops; Inductor cannot do materially better through the current scheduler-fusion classes because the required two fp16 reads, one fp16 write, and 2048-wide exp/reduction dominate this repro. The fix class is BANDWIDTH_BOUND: treat this artifact as diagnosis-only unless the measured Triton row kernel beats both required compile configs on the same full scope.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_4fb4f9a0bf43"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

DEFAULT_COLS = 2048

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
def _add_softmax_rows_kernel(
    bmm_ptr,
    bias_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    bias_s1: tl.constexpr,
    bias_s2: tl.constexpr,
    bias_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    head = row // q_len
    query = row - head * q_len
    cols = tl.arange(0, block_k)
    mask = cols < k_len

    bmm_offsets = head * bmm_s0 + query * bmm_s1 + cols * bmm_s2
    bias_offsets = head * bias_s1 + query * bias_s2 + cols * bias_s3

    bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0)
    bias_vals = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0)

    # repro.py performs aten.add on fp16 tensors before converting to fp32.
    scores = (bmm_vals + bias_vals).to(tl.float16).to(tl.float32)
    scores = tl.where(mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=0)
    numer = tl.exp2((scores - row_max) * 1.4426950408889634)
    denom = tl.sum(numer, axis=0)
    probs = numer / denom

    out_offsets = head * out_s0 + query * out_s1 + cols * out_s2
    tl.store(out_ptr + out_offsets, probs, mask=mask)


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


def _output_shape(
    bmm_34: torch.Tensor,
    _shape_param_2,
) -> tuple[int, int, int]:
    if _shape_param_2 is None:
        return tuple(bmm_34.shape)
    return tuple(int(dim) for dim in _shape_param_2)


def _validate_inputs(
    bmm_34: torch.Tensor,
    add_48: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[int, int, int]:
    if not bmm_34.is_cuda or not add_48.is_cuda:
        raise RuntimeError("CUDA tensors are required")
    if bmm_34.dtype != torch.float16 or add_48.dtype != torch.float16:
        raise TypeError(f"expected fp16 inputs, got {bmm_34.dtype} and {add_48.dtype}")
    if bmm_34.ndim != 3 or add_48.ndim != 4:
        raise ValueError(f"expected ranks 3 and 4, got {bmm_34.ndim} and {add_48.ndim}")

    heads, q_len, k_len = (int(dim) for dim in bmm_34.shape)
    expected_4d = (1, heads, q_len, k_len)
    if tuple(add_48.shape) != expected_4d:
        raise ValueError(f"expected add_48 shape {expected_4d}, got {tuple(add_48.shape)}")

    if _shape_param_0 is not None and tuple(_shape_param_0) != expected_4d:
        raise ValueError(f"_shape_param_0 mismatch: {tuple(_shape_param_0)} vs {expected_4d}")
    if _shape_param_1 is not None and tuple(_shape_param_1) != expected_4d:
        raise ValueError(f"_shape_param_1 mismatch: {tuple(_shape_param_1)} vs {expected_4d}")

    out_shape = _output_shape(bmm_34, _shape_param_2)
    if out_shape != (heads, q_len, k_len):
        raise ValueError(f"expected output view {(heads, q_len, k_len)}, got {out_shape}")
    return heads, q_len, k_len


def _contiguous_3d_stride(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return (shape[1] * shape[2], shape[2], 1)


def _launch_oracle(
    bmm_34: torch.Tensor,
    add_48: torch.Tensor,
    out: torch.Tensor,
    *,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    heads, q_len, k_len = (int(dim) for dim in bmm_34.shape)
    if block_k < k_len or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {k_len}, got {block_k}")
    if out.shape != bmm_34.shape or out.dtype != torch.float16 or not out.is_cuda:
        raise ValueError("preallocated output must be CUDA fp16 with bmm_34 shape")

    _add_softmax_rows_kernel[(heads * q_len,)](
        bmm_34,
        add_48,
        out,
        bmm_s0=bmm_34.stride(0),
        bmm_s1=bmm_34.stride(1),
        bmm_s2=bmm_34.stride(2),
        bias_s1=add_48.stride(1),
        bias_s2=add_48.stride(2),
        bias_s3=add_48.stride(3),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        q_len=q_len,
        k_len=k_len,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


def oracle_online_softmax(
    bmm_34: torch.Tensor,
    add_48: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
    *,
    block_k: int | None = None,
    num_warps: int = 8,
) -> torch.Tensor:
    heads, q_len, k_len = _validate_inputs(
        bmm_34,
        add_48,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    )
    out_shape = (heads, q_len, k_len)
    out = torch.empty_strided(
        out_shape,
        _contiguous_3d_stride(out_shape),
        device=bmm_34.device,
        dtype=torch.float16,
    )
    actual_block_k = block_k if block_k is not None else triton.next_power_of_2(k_len)
    return _launch_oracle(
        bmm_34,
        add_48,
        out,
        block_k=actual_block_k,
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


def run_check(block_k: int | None, num_warps: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=1234)
    model = module.Repro().cuda()

    with torch.no_grad():
        ref = model(*inputs)
        got = oracle_online_softmax(*inputs, block_k=block_k, num_warps=num_warps)
        torch.cuda.synchronize()

    max_abs, max_rel = _max_diff(got, ref)
    shape_match = tuple(got.shape) == tuple(ref.shape)
    dtype_match = got.dtype == ref.dtype
    stride_match = got.stride() == ref.stride()
    value_match = torch.allclose(got.float(), ref.float(), rtol=5e-3, atol=5e-4)
    ok = shape_match and dtype_match and stride_match and value_match

    print(
        "check full-scope T5 additive softmax: "
        f"shape={tuple(got.shape)} dtype={got.dtype} stride={got.stride()} "
        f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
        f"shape_match={shape_match} dtype_match={dtype_match} "
        f"stride_match={stride_match} allclose={value_match}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def run_bench(
    block_k: int | None,
    num_warps: int,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321)
    bmm_34, add_48 = inputs[:2]
    _validate_inputs(*inputs)
    heads, q_len, k_len = (int(dim) for dim in bmm_34.shape)
    out_shape = (heads, q_len, k_len)
    out = torch.empty_strided(
        out_shape,
        _contiguous_3d_stride(out_shape),
        device=bmm_34.device,
        dtype=torch.float16,
    )
    actual_block_k = block_k if block_k is not None else triton.next_power_of_2(k_len)
    traffic_bytes = heads * q_len * k_len * 2 * 3

    print(
        "oracle shape: "
        f"bmm_34=f16[{heads},{q_len},{k_len}] "
        f"add_48=f16[1,{heads},{q_len},{k_len}] "
        f"out=f16[{heads},{q_len},{k_len}] stride={out.stride()}"
    )
    print(f"single-kernel logical traffic: {traffic_bytes / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(
                bmm_34,
                add_48,
                out,
                block_k=actual_block_k,
                num_warps=num_warps,
            ),
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = traffic_bytes / (oracle_us * 1e-6) / 1e12
    print(
        "oracle full-scope additive softmax: "
        f"{oracle_us:.3f} us ({oracle_bw:.3f} TB/s logical bytes)"
    )
    print(f"oracle_us={oracle_us:.3f}")

    if no_compile:
        return

    print("CUDA graph replay timings cover the same repro.py add, softmax, cast, expand, and view")
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
    parser.add_argument("--block-k", type=int, default=None, help="Triton row tile size")
    parser.add_argument("--num-warps", type=int, default=8, help="Triton warps per row")
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
        ok = run_check(block_k=args.block_k, num_warps=args.num_warps)
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench(
            block_k=args.block_k,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
