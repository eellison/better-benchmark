"""Full-scope Triton oracle for amax_sum_cd5b9c2339af.

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the
complete MT5 attention softmax/dropout region from repro.py, including the
view of f32[192,128,128] scores to [32,6,128,128], the f32 position-bias add,
stable last-dimension softmax, Inductor stateless RNG dropout using
tl.rand(seed, flat_offset), dropout scaling, expand/view, and final
non-contiguous [192,128,128] permuted output stride. It differs from Inductor
by using a shape-specialized row-blocked online-softmax/dropout Triton template
that groups eight K=128 rows per program while writing the final permuted
stride directly. Inductor can fuse the graph today, but its generic
persistent-reduction schedule/autotune path does not select this small-K
attention-softmax/dropout row-blocked template under the required configs. The
Inductor change that would fix it is a dedicated or expanded online-softmax
dropout/layout template for K=128 attention rows that includes this row-grouped
schedule and gates it by shape.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl


REPRO_ID = "amax_sum_cd5b9c2339af"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
HEADS = 6
Q_LEN = 128
K_LEN = 128
BH = BATCH * HEADS
N_ROWS = BH * Q_LEN
BMM_SHAPE = (BH, Q_LEN, K_LEN)
ADD_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
OUT_SHAPE = BMM_SHAPE
OUT_STRIDE = (Q_LEN * K_LEN, 1, K_LEN)
SEED_INDEX = 79
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
HISTORICAL_BEST_COMPILE_US = 19.36000026762485
CLASSIFICATION = "NEW_PATTERN"

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
def _softmax_dropout_kernel(
    bmm_ptr,
    add_ptr,
    seeds_ptr,
    out_ptr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_k)
    row_mask = rows < 24576
    col_mask = cols < 128
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * 128 + cols[None, :]

    bmm_vals = tl.load(bmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    add_vals = tl.load(add_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scores = bmm_vals + add_vals
    scores = tl.where(mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    numer = tl.exp(scores - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)

    seed = tl.load(seeds_ptr + 79)
    random_vals = tl.rand(seed, offsets.to(tl.uint32))
    keep = (random_vals > 0.1).to(tl.float32)
    out_vals = (numer / denom[:, None]) * keep * 1.1111111111111112

    tl.store(out_ptr + offsets, out_vals, mask=mask)


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _make_inputs(module: Any, seed: int) -> tuple[Any, ...]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = module.make_inputs()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)


def _validate_shape_param(name: str, actual: object, expected: tuple[int, ...]) -> None:
    if tuple(int(dim) for dim in actual) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(actual)}")


def _validate_inputs(
    bmm_46: torch.Tensor,
    add_70: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
) -> None:
    if not (bmm_46.is_cuda and add_70.is_cuda and inductor_seeds.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if bmm_46.dtype != torch.float32 or add_70.dtype != torch.float32:
        raise TypeError(f"expected fp32 score inputs, got {bmm_46.dtype} and {add_70.dtype}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"expected int64 inductor_seeds, got {inductor_seeds.dtype}")
    if tuple(bmm_46.shape) != BMM_SHAPE:
        raise ValueError(f"unexpected bmm_46 shape: {tuple(bmm_46.shape)}")
    if tuple(add_70.shape) != ADD_SHAPE:
        raise ValueError(f"unexpected add_70 shape: {tuple(add_70.shape)}")
    if tuple(inductor_seeds.shape) != (84,):
        raise ValueError(f"unexpected inductor_seeds shape: {tuple(inductor_seeds.shape)}")
    if tuple(bmm_46.stride()) != (Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(f"unexpected bmm_46 stride: {tuple(bmm_46.stride())}")
    if tuple(add_70.stride()) != (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(f"unexpected add_70 stride: {tuple(add_70.stride())}")
    _validate_shape_param("_shape_param_0", _shape_param_0, ADD_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, ADD_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


def _make_output(device: torch.device) -> torch.Tensor:
    return torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=device,
        dtype=torch.float32,
    )


def _launch_oracle(
    bmm_46: torch.Tensor,
    add_70: torch.Tensor,
    inductor_seeds: torch.Tensor,
    out: torch.Tensor,
    *,
    block_m: int,
    num_warps: int,
) -> torch.Tensor:
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError(f"block_m must be a positive power of two, got {block_m}")
    if tuple(out.shape) != OUT_SHAPE or tuple(out.stride()) != OUT_STRIDE:
        raise ValueError(f"output must have shape {OUT_SHAPE} and stride {OUT_STRIDE}")
    if out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError("output must be CUDA fp32")

    _softmax_dropout_kernel[(triton.cdiv(N_ROWS, block_m),)](
        bmm_46,
        add_70,
        inductor_seeds,
        out,
        block_m=block_m,
        block_k=K_LEN,
        num_warps=num_warps,
    )
    return out


def oracle_online_softmax(
    bmm_46: torch.Tensor,
    add_70: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    *,
    block_m: int = 8,
    num_warps: int = 4,
) -> torch.Tensor:
    _validate_inputs(
        bmm_46,
        add_70,
        inductor_seeds,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    )
    out = _make_output(bmm_46.device)
    return _launch_oracle(
        bmm_46,
        add_70,
        inductor_seeds,
        out,
        block_m=block_m,
        num_warps=num_warps,
    )


def oracle_forward(inputs: tuple[Any, ...]) -> torch.Tensor:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    return oracle_online_softmax(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    max_abs = torch.nan_to_num(diff, nan=0.0).max().item()
    rel = diff / expected_f32.abs().clamp_min(1e-8)
    max_rel = torch.nan_to_num(rel, nan=0.0).max().item()
    return max_abs, max_rel


def _bench_cuda_graph(fn: object, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(rep):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _compile_with_config(
    module: Any,
    inputs: tuple[Any, ...],
    config: dict[str, object],
    warmup: int,
) -> object:
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


def run_check(block_m: int, num_warps: int, rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=1234)
    _validate_inputs(*inputs)
    # Eager prims.inductor_random ignores the seed tensor and calls torch.rand,
    # while the compiled target uses tl.rand(seed, flat_offset).  Compare against
    # the compiled repro so the stochastic mask is the one this oracle implements.
    compiled = _compile_with_config(module, inputs, {}, warmup=1)

    with torch.no_grad():
        expected = _as_tuple(compiled(*inputs))
        actual = _as_tuple(
            oracle_online_softmax(*inputs, block_m=block_m, num_warps=num_warps)
        )
        torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(f"output arity mismatch: oracle={len(actual)} ref={len(expected)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for idx, (got_item, ref_item) in enumerate(zip(actual, expected)):
        if not isinstance(got_item, torch.Tensor) or not isinstance(ref_item, torch.Tensor):
            item_ok = got_item == ref_item
            print(f"output[{idx}]: non-tensor equal={item_ok}")
            ok = ok and bool(item_ok)
            continue

        metadata_ok = (
            got_item.shape == ref_item.shape
            and got_item.dtype == ref_item.dtype
            and got_item.stride() == ref_item.stride()
        )
        value_ok = torch.allclose(
            got_item.float(),
            ref_item.float(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        max_abs, max_rel = _max_diff(got_item, ref_item)
        item_ok = metadata_ok and value_ok
        ok = ok and bool(item_ok)
        print(
            f"output[{idx}]: shape={list(got_item.shape)} dtype={got_item.dtype} "
            f"stride={list(got_item.stride())} ref_stride={list(ref_item.stride())} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} metadata={metadata_ok}"
        )

    print(
        "check compared against compiled full Repro.forward return value, including "
        "f32 add, stable softmax, Inductor tl.rand-compatible dropout, "
        "dropout scale, expand/view, and final permuted stride"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_one_oracle(
    inputs: tuple[Any, ...],
    *,
    block_m: int,
    num_warps: int,
    warmup: int,
    rep: int,
) -> float:
    bmm_46, add_70, inductor_seeds, *_ = inputs
    out = _make_output(bmm_46.device)
    return _bench_cuda_graph(
        lambda: _launch_oracle(
            bmm_46,
            add_70,
            inductor_seeds,
            out,
            block_m=block_m,
            num_warps=num_warps,
        ),
        warmup=warmup,
        rep=rep,
    )


def run_bench(
    block_m: int,
    num_warps: int,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321)
    _validate_inputs(*inputs)

    score_elems = N_ROWS * K_LEN
    logical_bytes = score_elems * 4 * 3
    print(
        "oracle shape: "
        f"bmm_46=f32{BMM_SHAPE} stride={(Q_LEN * K_LEN, K_LEN, 1)} "
        f"add_70=f32{ADD_SHAPE} out=f32{OUT_SHAPE} stride={OUT_STRIDE}"
    )
    print(
        "oracle tiling: "
        f"rows={N_ROWS} k={K_LEN} block_m={block_m} num_warps={num_warps} "
        f"logical_read_write_bytes={logical_bytes / 1e6:.1f} MB"
    )

    with torch.no_grad():
        oracle_us = _bench_one_oracle(
            inputs,
            block_m=block_m,
            num_warps=num_warps,
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = logical_bytes / (oracle_us * 1e-6) / 1e12
    print(
        "oracle full-scope softmax/dropout/layout: "
        f"{oracle_us:.3f} us ({oracle_bw:.3f} TB/s logical bytes)"
    )
    print(f"oracle_us={oracle_us:.3f}")

    compile_results: list[dict[str, object]] = []
    if not no_compile:
        print(
            "torch.compile timings cover the same repro.py view, add, "
            "softmax, Inductor RNG dropout, scale, expand/view, and final permute"
        )
        holder: list[Any] = [None]
        for label, config in COMPILE_CONFIGS:
            try:
                compiled = _compile_with_config(module, inputs, config, warmup)
                us = _bench_cuda_graph(
                    lambda: holder.__setitem__(0, compiled(*inputs)),
                    warmup=warmup,
                    rep=rep,
                )
                compile_results.append({"label": label, "us": us})
                print(f"torch.compile {label}: {us:.3f} us")
            except Exception as exc:
                compile_results.append({"label": label, "error": str(exc)})
                print(f"torch.compile {label}: FAILED ({exc})")

    successful_compile_us = [
        float(result["us"]) for result in compile_results if "us" in result
    ]
    best_required_compile_us = (
        min(successful_compile_us) if successful_compile_us else None
    )
    required_compile_gate = (
        len(successful_compile_us) == len(COMPILE_CONFIGS)
        and best_required_compile_us is not None
        and oracle_us < best_required_compile_us
    )
    historical_gate = oracle_us < HISTORICAL_BEST_COMPILE_US
    true_floor = bool(required_compile_gate and historical_gate)

    if best_required_compile_us is not None:
        print(f"best_required_compile_us={best_required_compile_us:.3f}")
    print(f"historical_best_compile_us={HISTORICAL_BEST_COMPILE_US:.3f}")
    print(f"beats_required_compile={required_compile_gate}")
    print(f"beats_historical_best={historical_gate}")
    print(f"true_floor={true_floor}")
    if not true_floor:
        print("diagnosis_only: oracle did not beat both required configs and historical best")

    result = {
        "repro_id": REPRO_ID,
        "classification": CLASSIFICATION,
        "oracle_us": oracle_us,
        "compile_results": compile_results,
        "best_required_compile_us": best_required_compile_us,
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "beats_required_compile": required_compile_gate,
        "beats_historical_best": historical_gate,
        "true_floor": true_floor,
        "status": "GOOD" if true_floor else "DIAGNOSIS_ONLY",
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="benchmark repetitions")
    parser.add_argument("--block-m", type=int, default=8, help="rows per Triton program")
    parser.add_argument("--num-warps", type=int, default=4, help="Triton warps per program")
    parser.add_argument("--rtol", type=float, default=5e-5)
    parser.add_argument("--atol", type=float, default=5e-6)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile baselines for the requested configs",
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(
        block_m=args.block_m,
        num_warps=args.num_warps,
        rtol=args.rtol,
        atol=args.atol,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            block_m=args.block_m,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
