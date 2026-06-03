"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full T5 inference attention softmax returned by Repro.forward, including the view/expand/view layout contract, by proving the generated iota/ge/where mask always contributes a zero fp16 bias and folding it before the fp32 row softmax writes the contiguous [8, 2048, 2048] fp16 output, whereas Inductor lowers the decomposed view/full/iota/ge/where/add/cast/amax/sub/exp/sum/div/cast/expand/view graph as generic pointwise mask/bias work adjacent to the softmax reductions; Inductor cannot do this today because its scheduler/codegen simplifier does not prove shape-derived predicates such as arange(2048) >= 0 are tautologically true through expand/where and delete the resulting zero additive bias before reduction scheduling; the fix is ALGEBRAIC_ELIMINATION: add shape-aware predicate/value simplification that canonicalizes always-true structured masks and removes zero-bias attention additions before existing softmax codegen."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl


REPRO_ID = "amax_sum_7493058b895f"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 1
N_HEADS = 8
Q_LEN = 2048
K_LEN = 2048
N_ROWS = BATCH * N_HEADS * Q_LEN
OUT_SHAPE = (N_HEADS, Q_LEN, K_LEN)
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)


@triton.jit
def _attention_softmax_kernel(
    bmm_ptr,
    out_ptr,
    k_len: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_k)
    mask = cols < k_len
    offsets = row * k_len + cols

    # The captured iota/ge/where path is true for every key position and adds 0.
    scores = tl.load(bmm_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(scores, axis=0)
    numer = tl.exp(scores - row_max)
    denom = tl.sum(numer, axis=0)
    out = numer / denom

    tl.store(out_ptr + offsets, out, mask=mask)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_inputs(module, seed: int) -> tuple[object, ...]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = module.make_inputs()
    return tuple(
        x.cuda() if isinstance(x, torch.Tensor) and not x.is_cuda else x
        for x in inputs
    )


def _validate_shape_param(name: str, actual, expected: tuple[int, ...]) -> None:
    if tuple(int(dim) for dim in actual) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(actual)}")


def _validate_inputs(
    bmm_14: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> None:
    if not bmm_14.is_cuda:
        raise RuntimeError("CUDA input is required")
    if bmm_14.dtype != torch.float16:
        raise TypeError(f"expected bmm_14 dtype torch.float16, got {bmm_14.dtype}")
    if tuple(bmm_14.shape) != OUT_SHAPE:
        raise ValueError(f"unexpected bmm_14 shape: {tuple(bmm_14.shape)}")
    if tuple(bmm_14.stride()) != OUT_STRIDE:
        raise ValueError(f"unexpected bmm_14 stride: {tuple(bmm_14.stride())}")

    _validate_shape_param("_shape_param_0", _shape_param_0, (BATCH, N_HEADS, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_1", _shape_param_1, (BATCH, -1, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_2", _shape_param_2, (BATCH, N_HEADS, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_3", _shape_param_3, OUT_SHAPE)


def _launch_oracle(
    bmm_14: torch.Tensor,
    out: torch.Tensor,
    *,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    _attention_softmax_kernel[(N_ROWS,)](
        bmm_14,
        out,
        k_len=K_LEN,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


def oracle_full_attention_softmax(
    bmm_14: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    *,
    out: torch.Tensor | None = None,
    block_k: int = K_LEN,
    num_warps: int = 8,
) -> torch.Tensor:
    _validate_inputs(
        bmm_14,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )
    if block_k < K_LEN or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {K_LEN}, got {block_k}")
    if out is None:
        out = torch.empty_strided(
            OUT_SHAPE,
            OUT_STRIDE,
            device=bmm_14.device,
            dtype=torch.float16,
        )
    else:
        if tuple(out.shape) != OUT_SHAPE or tuple(out.stride()) != OUT_STRIDE:
            raise ValueError(f"unexpected output layout: shape={tuple(out.shape)} stride={tuple(out.stride())}")
        if out.dtype != torch.float16 or out.device != bmm_14.device:
            raise ValueError(f"unexpected output dtype/device: {out.dtype} {out.device}")

    return _launch_oracle(bmm_14, out, block_k=block_k, num_warps=num_warps)


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, torch.Tensor):
        return (value,)
    raise TypeError(f"unexpected Repro.forward return type: {type(value)!r}")


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    rel = diff / expected_f.abs().clamp_min(1e-8)
    return diff.max().item(), rel.max().item()


def run_check(
    *,
    seed: int,
    block_k: int,
    num_warps: int,
    rtol: float,
    atol: float,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=seed)
    model = module.Repro().cuda()

    with torch.no_grad():
        expected = _as_tuple(model(*inputs))
        actual = _as_tuple(
            oracle_full_attention_softmax(
                *inputs,
                block_k=block_k,
                num_warps=num_warps,
            )
        )
        torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(f"output tuple length mismatch: got {len(actual)}, expected {len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        shape_ok = tuple(got.shape) == tuple(ref.shape)
        dtype_ok = got.dtype == ref.dtype
        stride_ok = tuple(got.stride()) == tuple(ref.stride())
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            values_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs = float("nan")
            max_rel = float("nan")
            values_ok = False
        output_ok = shape_ok and dtype_ok and stride_ok and values_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={tuple(got.shape)} ref_shape={tuple(ref.shape)} "
            f"shape_ok={shape_ok} dtype={got.dtype} ref_dtype={ref.dtype} dtype_ok={dtype_ok} "
            f"stride={tuple(got.stride())} ref_stride={tuple(ref.stride())} stride_ok={stride_ok} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} values_ok={values_ok}"
        )

    print("check compared every output against full Repro.forward")
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_cuda(fn, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def run_bench(*, seed: int, block_k: int, num_warps: int, warmup: int, rep: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=seed)
    bmm_14 = inputs[0]
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_14.device,
        dtype=torch.float16,
    )

    dense_elems = N_ROWS * K_LEN
    traffic_bytes = dense_elems * 2 * 2
    print(
        f"oracle shape: bmm=f16{OUT_SHAPE}, rows={N_ROWS}, reduction={K_LEN}, "
        f"output_stride={OUT_STRIDE}"
    )
    print(f"single-kernel logical read+write traffic: {traffic_bytes / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda(
            lambda: _launch_oracle(
                bmm_14,
                out,
                block_k=block_k,
                num_warps=num_warps,
            ),
            warmup=warmup,
            rep=rep,
        )

    bandwidth_tbps = traffic_bytes / (oracle_us * 1e-6) / 1e12
    print(f"oracle full attention softmax: {oracle_us:.3f} us ({bandwidth_tbps:.3f} TB/s logical)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=25, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="benchmark repetitions")
    parser.add_argument("--seed", type=int, default=1234, help="input generation seed")
    parser.add_argument("--block-k", type=int, default=K_LEN, help="Triton row tile size")
    parser.add_argument("--num-warps", type=int, default=8, help="Triton warps per row")
    parser.add_argument("--rtol", type=float, default=1e-2, help="correctness relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-5, help="correctness absolute tolerance")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.block_k < K_LEN:
        args.block_k = triton.next_power_of_2(K_LEN)

    if args.check:
        ok = run_check(
            seed=args.seed,
            block_k=args.block_k,
            num_warps=args.num_warps,
            rtol=args.rtol,
            atol=args.atol,
        )
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench(
            seed=args.seed,
            block_k=args.block_k,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
