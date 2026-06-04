"""Full-scope Triton oracle for sum_50185c527fec.

Gap diagnosis (classification: BANDWIDTH_BOUND): the repro is the Swin
attention backward row update

    out[b, h, i, j] = div[b, h, i, j] * (bmm[b, h, i, j]
                                        - sum_k bmm[b, h, i, k] * div[b, h, i, k])

with the output reshaped to contiguous f32[32768, 49, 49]. This oracle consumes
the same bmm tensor, the same padded-stride div tensor, and the same shape
parameters as repro.py, then computes the full multiply, row reduction,
broadcast FMA, and final contiguous output in one Triton kernel. It does not
time a reduction-only subset.

Inductor already emits one fused persistent-reduction Triton kernel for this
scope, so there is no missing high-level fusion, scatter-reduce, split-K,
algebraic-elimination, or recomputation transformation to unlock. The only
possible improvement is a small fixed-shape tiling/autotune choice for this
49-wide row reduction; if that does not beat both required local compile
configs and the historical best compile time, this oracle is diagnosis-only and
the repro should be closed as bandwidth-bound/already at floor.
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


REPRO_ID = "sum_50185c527fec"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8192
HEADS = 4
Q = 49
K = 49
BH = BATCH * HEADS
ROWS = BH * Q
OUT_SHAPE = (BH, Q, K)
DIV_SHAPE = (BATCH, HEADS, Q, K)
BMM_STRIDE = (Q * K, K, 1)
DIV_STRIDE = (HEADS * 2432, 2432, K, 1)
R_BLOCK = triton.next_power_of_2(K)
DEFAULT_ROWS_PER_BLOCK = 16
DEFAULT_ROWS_PER_BLOCK_CANDIDATES = (4, 8, 16, 32)
HISTORICAL_BEST_COMPILE_US = 156.63999319076538
CLASSIFICATION = "BANDWIDTH_BOUND"

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
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
def _row_reduce_fma_kernel(
    bmm_ptr,
    div_ptr,
    out_ptr,
    ROWS_: tl.constexpr,
    Q_: tl.constexpr,
    K_: tl.constexpr,
    DIV_BH_STRIDE_: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    R_BLOCK_: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)[:, None]
    cols = tl.arange(0, R_BLOCK_)[None, :]
    row_mask = rows < ROWS_
    col_mask = cols < K_
    mask = row_mask & col_mask

    q_index = rows % Q_
    bh_index = rows // Q_
    bmm_offsets = rows * K_ + cols
    div_offsets = bh_index * DIV_BH_STRIDE_ + q_index * K_ + cols

    bmm = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    div = tl.load(div_ptr + div_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    product = bmm * div
    row_sum = tl.sum(tl.where(col_mask, product, 0.0), axis=1)[:, None].to(tl.float32)
    out = tl.fma(-div, row_sum, product)
    tl.store(out_ptr + bmm_offsets, out, mask=mask)


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
    return tuple(x.cuda() if isinstance(x, torch.Tensor) and x.device.type != "cuda" else x for x in module.make_inputs())


def _num_warps(rows_per_block: int) -> int:
    if rows_per_block <= 4:
        return 4
    return 8


def _validate_inputs(
    bmm_137: torch.Tensor,
    div_1: torch.Tensor,
    shape_param_0: object,
    shape_param_1: object,
) -> None:
    assert tuple(bmm_137.shape) == OUT_SHAPE
    assert tuple(div_1.shape) == DIV_SHAPE
    assert tuple(shape_param_0) == DIV_SHAPE
    assert tuple(shape_param_1) == OUT_SHAPE
    assert bmm_137.dtype == torch.float32
    assert div_1.dtype == torch.float32
    assert bmm_137.stride() == BMM_STRIDE
    assert div_1.stride() == DIV_STRIDE


def _oracle_into(
    bmm_137: torch.Tensor,
    div_1: torch.Tensor,
    out: torch.Tensor,
    rows_per_block: int,
) -> torch.Tensor:
    assert out.shape == OUT_SHAPE
    assert out.dtype == torch.float32
    assert out.stride() == BMM_STRIDE

    grid = (triton.cdiv(ROWS, rows_per_block),)
    _row_reduce_fma_kernel[grid](
        bmm_137,
        div_1,
        out,
        ROWS_=ROWS,
        Q_=Q,
        K_=K,
        DIV_BH_STRIDE_=DIV_STRIDE[1],
        ROWS_PER_BLOCK=rows_per_block,
        R_BLOCK_=R_BLOCK,
        num_warps=_num_warps(rows_per_block),
    )
    return out


def oracle_fused(
    bmm_137: torch.Tensor,
    div_1: torch.Tensor,
    shape_param_0: object,
    shape_param_1: object,
    rows_per_block: int = DEFAULT_ROWS_PER_BLOCK,
) -> torch.Tensor:
    _validate_inputs(bmm_137, div_1, shape_param_0, shape_param_1)
    out = torch.empty(OUT_SHAPE, device=bmm_137.device, dtype=torch.float32)
    return _oracle_into(bmm_137, div_1, out, rows_per_block)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        out = model(*inputs)
    return (out,) if isinstance(out, torch.Tensor) else tuple(out)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float, rows_per_block: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_outputs(inputs)
        actual_raw = oracle_fused(*inputs, rows_per_block=rows_per_block)
        actual = (actual_raw,) if isinstance(actual_raw, torch.Tensor) else tuple(actual_raw)
        torch.cuda.synchronize()

    if len(actual) != len(ref):
        print(f"output_count_match=False actual={len(actual)} expected={len(ref)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for idx, (actual_tensor, ref_tensor) in enumerate(zip(actual, ref, strict=True)):
        max_abs, max_rel = _max_diff(actual_tensor, ref_tensor)
        output_ok = torch.allclose(actual_tensor.float(), ref_tensor.float(), rtol=rtol, atol=atol)
        shape_ok = actual_tensor.shape == ref_tensor.shape
        stride_ok = actual_tensor.stride() == ref_tensor.stride()
        dtype_ok = actual_tensor.dtype == ref_tensor.dtype
        tensor_ok = output_ok and shape_ok and stride_ok and dtype_ok
        ok = ok and tensor_ok
        print(
            f"output[{idx}]: shape={list(actual_tensor.shape)} stride={actual_tensor.stride()} "
            f"dtype={actual_tensor.dtype} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={output_ok} shape_match={shape_ok} stride_match={stride_ok} "
            f"dtype_match={dtype_ok}"
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


def _capture_graph_or_fallback(fn):
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            return lambda: graph.replay(), "cudagraph"
        except Exception:
            return fn, "direct"


def _time_callable(fn, warmup: int, rep: int) -> float:
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def _parse_rows_per_block_candidates(raw: str | None) -> tuple[int, ...]:
    if raw is None or raw.strip() == "":
        return DEFAULT_ROWS_PER_BLOCK_CANDIDATES
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--rows-per-block-candidates produced no candidates")
    for value in values:
        if value <= 0 or value & (value - 1):
            raise ValueError("rows-per-block candidates must be positive powers of two")
    return values


def run_bench(rep: int, warmup: int, no_compile: bool, candidates: tuple[int, ...]) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    bmm_137, div_1, shape_param_0, shape_param_1 = inputs
    _validate_inputs(bmm_137, div_1, shape_param_0, shape_param_1)

    oracle_times: list[tuple[int, float, str]] = []
    with torch.no_grad():
        for rows_per_block in candidates:
            out = torch.empty(OUT_SHAPE, device=bmm_137.device, dtype=torch.float32)
            _oracle_into(bmm_137, div_1, out, rows_per_block)
            torch.cuda.synchronize()
            timed, timing_mode = _capture_graph_or_fallback(
                lambda rows_per_block=rows_per_block, out=out: _oracle_into(
                    bmm_137,
                    div_1,
                    out,
                    rows_per_block,
                )
            )
            oracle_us = _time_callable(timed, warmup=warmup, rep=rep)
            oracle_times.append((rows_per_block, oracle_us, timing_mode))
            print(
                f"oracle_fused rows_per_block={rows_per_block}: "
                f"{oracle_us:.3f} us ({timing_mode})"
            )

    best_rows_per_block, best_oracle_us, best_timing_mode = min(oracle_times, key=lambda item: item[1])
    print(
        f"oracle_fused best full-scope row reduction + FMA: "
        f"{best_oracle_us:.3f} us (rows_per_block={best_rows_per_block}, {best_timing_mode})"
    )

    compile_times: list[tuple[str, float, str]] = []
    if not no_compile:
        module = _load_repro_module()
        for label, config in COMPILE_CONFIGS:
            model = module.Repro().cuda()
            compiled = _compile_with_config(model, inputs, config)
            timed, timing_mode = _capture_graph_or_fallback(lambda compiled=compiled: compiled(*inputs))
            compiled_us = _time_callable(timed, warmup=warmup, rep=rep)
            compile_times.append((label, compiled_us, timing_mode))
            print(f"torch.compile {label}: {compiled_us:.3f} us ({timing_mode})")

    local_floor_ok = bool(compile_times) and all(best_oracle_us < compiled_us for _, compiled_us, _ in compile_times)
    historical_floor_ok = best_oracle_us < HISTORICAL_BEST_COMPILE_US
    true_floor = local_floor_ok and historical_floor_ok
    print(f"historical_best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"true_floor: {'yes' if true_floor else 'no'}")
    if not true_floor:
        print("diagnosis_only: oracle does not beat every required local compile config and historical best")

    result: dict[str, object] = {
        "repro_id": REPRO_ID,
        "classification": CLASSIFICATION,
        "oracle_us": best_oracle_us,
        "best_rows_per_block": best_rows_per_block,
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "true_floor": true_floor,
        "status": "TRUE_FLOOR" if true_floor else "DIAGNOSIS_ONLY",
    }
    for label, compiled_us, _ in compile_times:
        result[f"compile_{label}_us"] = compiled_us
    if compile_times:
        result["best_required_compile_us"] = min(compiled_us for _, compiled_us, _ in compile_times)
        result["ratio_compile_best_to_oracle"] = result["best_required_compile_us"] / best_oracle_us
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rows-per-block", type=int, default=DEFAULT_ROWS_PER_BLOCK)
    parser.add_argument(
        "--rows-per-block-candidates",
        default=None,
        help="comma-separated positive powers of two for --bench; defaults to 4,8,16,32",
    )
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    if args.rows_per_block <= 0 or args.rows_per_block & (args.rows_per_block - 1):
        raise ValueError("--rows-per-block must be a positive power of two")
    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol, rows_per_block=args.rows_per_block):
        sys.exit(1)
    if args.bench:
        candidates = _parse_rows_per_block_candidates(args.rows_per_block_candidates)
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile, candidates=candidates)


if __name__ == "__main__":
    with torch.no_grad():
        main()
