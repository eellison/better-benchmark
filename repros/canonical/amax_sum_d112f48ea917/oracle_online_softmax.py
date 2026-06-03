"""
Gap diagnosis (classification: NEW_PATTERN): this oracle covers the full
Reformer LSH equality-masked softmax materialization from repro.py, including
the rotated/current bucket concatenation, the broadcasted int64 equality mask,
the scalar replacement through aten.where, the stable logsumexp softmax over
the last dimension, and the final contiguous [6144, 64, 128] view. It differs
from Inductor by using one Triton program for several adjacent w rows of the
same (batch, head, q) group, so the 128-element comparison vector is loaded
once per tile instead of being materialized or redundantly reloaded around the
generic amax/sum lowering. Inductor cannot do this today because the mask is
expressed as view/slice/cat/unsqueeze/ne feeding a reduction pattern, and the
scheduler does not canonicalize that structured Reformer predicate into a
single masked-softmax row template with comparison-vector reuse. The fix class
is NEW_PATTERN: add a lowering for this Reformer LSH masked softmax that
recomputes the cheap predicate inside the online-softmax kernel and fuses the
epilogue view.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_d112f48ea917"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
HEADS = 12
Q_LEN = 64
W_LEN = 64
C_LEN = 128
FLAT_BHQ = BATCH * HEADS * Q_LEN
OUT_SHAPE = (FLAT_BHQ, W_LEN, C_LEN)
OUT_STRIDE = (W_LEN * C_LEN, C_LEN, 1)

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
def _reformer_masked_softmax_kernel(
    remainder_ptr,
    bmm_ptr,
    scalar_ptr,
    out_ptr,
    rem_s0: tl.constexpr,
    rem_s1: tl.constexpr,
    rem_s2: tl.constexpr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    w_len: tl.constexpr,
    c_len: tl.constexpr,
    block_w: tl.constexpr,
    block_c: tl.constexpr,
):
    group = tl.program_id(0)
    w_block = tl.program_id(1)

    batch = group // (heads * q_len)
    rem_group = group - batch * (heads * q_len)
    head = rem_group // q_len
    q = rem_group - head * q_len
    q_prev = tl.where(q == 0, q_len - 1, q - 1)

    w_offsets = w_block * block_w + tl.arange(0, block_w)
    c_offsets = tl.arange(0, block_c)
    w_mask = w_offsets < w_len
    c_mask = c_offsets < c_len

    rem_base = batch * rem_s0 + head * rem_s1
    left_vals = tl.load(
        remainder_ptr + rem_base + (q * w_len + w_offsets) * rem_s2,
        mask=w_mask,
        other=0,
    )

    right_q = tl.where(c_offsets < w_len, q_prev, q)
    right_w = tl.where(c_offsets < w_len, c_offsets, c_offsets - w_len)
    right_vals = tl.load(
        remainder_ptr + rem_base + (right_q * w_len + right_w) * rem_s2,
        mask=c_mask,
        other=0,
    )

    bmm_offsets = (
        group * bmm_s0
        + w_offsets[:, None] * bmm_s1
        + c_offsets[None, :] * bmm_s2
    )
    valid = w_mask[:, None] & c_mask[None, :]
    bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=valid, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    neq = left_vals[:, None] != right_vals[None, :]
    scores = tl.where(neq, bmm_vals, scalar)
    scores = tl.where(valid, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    stable_max = tl.where(tl.abs(row_max) == float("inf"), 0.0, row_max)
    exp_scores = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
    denom = tl.sum(exp_scores, axis=1)
    out_vals = exp_scores / denom[:, None]

    out_offsets = (
        group * out_s0
        + w_offsets[:, None] * out_s1
        + c_offsets[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, out_vals, mask=valid)


@triton.jit
def _reformer_masked_softmax_serial_w_kernel(
    remainder_ptr,
    bmm_ptr,
    scalar_ptr,
    out_ptr,
    rem_s0: tl.constexpr,
    rem_s1: tl.constexpr,
    rem_s2: tl.constexpr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    w_len: tl.constexpr,
    c_len: tl.constexpr,
    block_c: tl.constexpr,
):
    group = tl.program_id(0)

    batch = group // (heads * q_len)
    rem_group = group - batch * (heads * q_len)
    head = rem_group // q_len
    q = rem_group - head * q_len
    q_prev = tl.where(q == 0, q_len - 1, q - 1)

    c_offsets = tl.arange(0, block_c)
    c_mask = c_offsets < c_len

    rem_base = batch * rem_s0 + head * rem_s1
    right_q = tl.where(c_offsets < w_len, q_prev, q)
    right_w = tl.where(c_offsets < w_len, c_offsets, c_offsets - w_len)
    right_vals = tl.load(
        remainder_ptr + rem_base + (right_q * w_len + right_w) * rem_s2,
        mask=c_mask,
        other=0,
    )
    scalar = tl.load(scalar_ptr).to(tl.float32)

    for w in tl.range(0, w_len):
        left_val = tl.load(remainder_ptr + rem_base + (q * w_len + w) * rem_s2)
        bmm_offsets = group * bmm_s0 + w * bmm_s1 + c_offsets * bmm_s2
        bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=c_mask, other=0.0).to(tl.float32)

        scores = tl.where(left_val != right_vals, bmm_vals, scalar)
        scores = tl.where(c_mask, scores, -float("inf"))
        row_max = tl.max(scores, axis=0)
        stable_max = tl.where(tl.abs(row_max) == float("inf"), 0.0, row_max)
        exp_scores = tl.exp2((scores - stable_max) * 1.4426950408889634)
        denom = tl.sum(exp_scores, axis=0)
        out_vals = exp_scores / denom

        out_offsets = group * out_s0 + w * out_s1 + c_offsets * out_s2
        tl.store(out_ptr + out_offsets, out_vals, mask=c_mask)


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
    if tuple(int(dim) for dim in actual) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(actual)}")


def _validate_inputs(
    remainder_1: torch.Tensor,
    bmm_1: torch.Tensor,
    arg5_1: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
    _shape_param_3=None,
) -> None:
    if not (remainder_1.is_cuda and bmm_1.is_cuda and arg5_1.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if remainder_1.dtype != torch.int64:
        raise TypeError(f"expected remainder_1 int64, got {remainder_1.dtype}")
    if bmm_1.dtype != torch.float32 or arg5_1.dtype != torch.float32:
        raise TypeError(f"expected fp32 score/scalar inputs, got {bmm_1.dtype} and {arg5_1.dtype}")
    if tuple(remainder_1.shape) != (BATCH, HEADS, Q_LEN * W_LEN):
        raise ValueError(f"unexpected remainder_1 shape: {tuple(remainder_1.shape)}")
    if tuple(bmm_1.shape) != OUT_SHAPE:
        raise ValueError(f"unexpected bmm_1 shape: {tuple(bmm_1.shape)}")
    if arg5_1.ndim != 0:
        raise ValueError(f"expected scalar arg5_1, got shape {tuple(arg5_1.shape)}")

    if _shape_param_0 is not None and tuple(_shape_param_0) != (BATCH, HEADS, -1, W_LEN):
        raise ValueError(f"_shape_param_0 mismatch: {tuple(_shape_param_0)}")
    _validate_shape_param("_shape_param_1", _shape_param_1, (BATCH, HEADS, Q_LEN, W_LEN, C_LEN))
    _validate_shape_param("_shape_param_2", _shape_param_2, (BATCH, HEADS, Q_LEN, W_LEN, C_LEN))
    _validate_shape_param("_shape_param_3", _shape_param_3, OUT_SHAPE)


def _launch_oracle(
    remainder_1: torch.Tensor,
    bmm_1: torch.Tensor,
    arg5_1: torch.Tensor,
    out: torch.Tensor,
    *,
    block_w: int,
    block_c: int,
    num_warps: int,
    serial_w: bool,
) -> torch.Tensor:
    if out.shape != OUT_SHAPE or out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError("preallocated output must be CUDA fp32 with shape [6144, 64, 128]")
    if out.stride() != OUT_STRIDE:
        raise ValueError(f"preallocated output stride must be {OUT_STRIDE}, got {out.stride()}")
    if block_c < C_LEN or block_c & (block_c - 1):
        raise ValueError(f"block_c must be a power of two >= {C_LEN}, got {block_c}")

    if serial_w:
        _reformer_masked_softmax_serial_w_kernel[(FLAT_BHQ,)](
            remainder_1,
            bmm_1,
            arg5_1,
            out,
            rem_s0=remainder_1.stride(0),
            rem_s1=remainder_1.stride(1),
            rem_s2=remainder_1.stride(2),
            bmm_s0=bmm_1.stride(0),
            bmm_s1=bmm_1.stride(1),
            bmm_s2=bmm_1.stride(2),
            out_s0=out.stride(0),
            out_s1=out.stride(1),
            out_s2=out.stride(2),
            heads=HEADS,
            q_len=Q_LEN,
            w_len=W_LEN,
            c_len=C_LEN,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out

    if block_w <= 0 or W_LEN % block_w != 0:
        raise ValueError(f"block_w must divide {W_LEN}, got {block_w}")
    grid = (FLAT_BHQ, triton.cdiv(W_LEN, block_w))
    _reformer_masked_softmax_kernel[grid](
        remainder_1,
        bmm_1,
        arg5_1,
        out,
        rem_s0=remainder_1.stride(0),
        rem_s1=remainder_1.stride(1),
        rem_s2=remainder_1.stride(2),
        bmm_s0=bmm_1.stride(0),
        bmm_s1=bmm_1.stride(1),
        bmm_s2=bmm_1.stride(2),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        heads=HEADS,
        q_len=Q_LEN,
        w_len=W_LEN,
        c_len=C_LEN,
        block_w=block_w,
        block_c=block_c,
        num_warps=num_warps,
    )
    return out


def oracle_online_softmax(
    remainder_1: torch.Tensor,
    bmm_1: torch.Tensor,
    arg5_1: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
    _shape_param_3=None,
    *,
    block_w: int = 64,
    block_c: int = 128,
    num_warps: int = 8,
    serial_w: bool = False,
) -> torch.Tensor:
    _validate_inputs(
        remainder_1,
        bmm_1,
        arg5_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_1.device,
        dtype=torch.float32,
    )
    return _launch_oracle(
        remainder_1,
        bmm_1,
        arg5_1,
        out,
        block_w=block_w,
        block_c=block_c,
        num_warps=num_warps,
        serial_w=serial_w,
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


def run_check(block_w: int, block_c: int, num_warps: int, serial_w: bool) -> bool:
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
            block_w=block_w,
            block_c=block_c,
            num_warps=num_warps,
            serial_w=serial_w,
        )
        torch.cuda.synchronize()

    max_abs, max_rel = _max_diff(got, ref)
    shape_match = tuple(got.shape) == tuple(ref.shape)
    dtype_match = got.dtype == ref.dtype
    stride_match = got.stride() == ref.stride()
    value_match = torch.allclose(got, ref, rtol=1e-5, atol=1e-6, equal_nan=True)
    ok = shape_match and dtype_match and stride_match and value_match

    print(
        "check full-scope Reformer LSH masked softmax: "
        f"shape={tuple(got.shape)} dtype={got.dtype} stride={got.stride()} "
        f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
        f"shape_match={shape_match} dtype_match={dtype_match} "
        f"stride_match={stride_match} allclose={value_match}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def run_bench(
    block_w: int,
    block_c: int,
    num_warps: int,
    warmup: int,
    rep: int,
    no_compile: bool,
    serial_w: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321)
    _validate_inputs(*inputs)
    remainder_1, bmm_1, arg5_1 = inputs[:3]
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_1.device,
        dtype=torch.float32,
    )

    score_bytes = FLAT_BHQ * W_LEN * C_LEN * 4
    output_bytes = score_bytes
    right_tiles = 1 if serial_w else triton.cdiv(W_LEN, block_w)
    tiled_right_bytes = FLAT_BHQ * right_tiles * C_LEN * 8
    left_bytes = FLAT_BHQ * W_LEN * 8
    logical_bytes = score_bytes + output_bytes + tiled_right_bytes + left_bytes

    print(
        "oracle shape: "
        f"remainder=i64[{BATCH},{HEADS},{Q_LEN * W_LEN}] "
        f"bmm=f32[{OUT_SHAPE[0]},{OUT_SHAPE[1]},{OUT_SHAPE[2]}] "
        f"scalar=f32[] out=f32{OUT_SHAPE} stride={OUT_STRIDE}"
    )
    print(
        "oracle tiling: "
        f"block_w={block_w} block_c={block_c} num_warps={num_warps} "
        f"serial_w={serial_w} "
        f"logical_tiled_bytes={logical_bytes / 1e6:.1f} MB"
    )

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(
                remainder_1,
                bmm_1,
                arg5_1,
                out,
                block_w=block_w,
                block_c=block_c,
                num_warps=num_warps,
                serial_w=serial_w,
            ),
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = logical_bytes / (oracle_us * 1e-6) / 1e12
    print(
        "oracle full-scope Reformer LSH masked softmax: "
        f"{oracle_us:.3f} us ({oracle_bw:.3f} TB/s logical tiled bytes)"
    )
    print(f"oracle_us={oracle_us:.3f}")

    if no_compile:
        return

    print(
        "CUDA graph replay timings cover the same repro.py mask assembly, "
        "where, stable logsumexp softmax, and output view"
    )
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
        valid_floor = len(compile_times) == len(COMPILE_CONFIGS) and oracle_us < min(
            us for _, us in compile_times
        )
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
    parser.add_argument("--block-w", type=int, default=64, help="number of w rows per Triton program")
    parser.add_argument("--block-c", type=int, default=128, help="last-dimension Triton tile size")
    parser.add_argument("--num-warps", type=int, default=8, help="Triton warps per program")
    parser.add_argument("--serial-w", action="store_true", help="loop over all w rows inside each group program")
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
            block_w=args.block_w,
            block_c=args.block_c,
            num_warps=args.num_warps,
            serial_w=args.serial_w,
        )
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench(
            block_w=args.block_w,
            block_c=args.block_c,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
            serial_w=args.serial_w,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
