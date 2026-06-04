"""
Full-scope Triton oracle for sum_f403c03c9650.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle consumes the same
two f32[131072, 288] tensors and shape parameters as repro.py, computes the
full sigmoid/SiLU-gradient pointwise producer, writes the backing storage for
the returned f32[288, 131072] transposed view with stride (1, 288), and
accumulates the returned f32[288] feature sum from the same pass. This differs
from a naive schedule by avoiding a second full read of the producer for the
sum, but it does not currently demonstrate an Inductor scheduling limitation:
the historical best compile time is already at the memory/exp throughput floor
for this full materialization plus reduction. The Inductor fix classification
is therefore BANDWIDTH_BOUND; no compiler change should be claimed unless a
full-scope Triton implementation beats both required local compile configs and
the historical best_compile_us gate.
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


REPRO_ID = "sum_f403c03c9650"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

M = 131072
N = 288
HISTORICAL_BEST_COMPILE_US = 87.90399879217148
DEFAULT_BLOCK_M = 256
DEFAULT_BLOCK_N = 32

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _pointwise_copy_reduce_partials_kernel(
    mm_ptr,
    addmm_ptr,
    layout_storage_ptr,
    partials_ptr,
    num_m_blocks: tl.constexpr,
    M_: tl.constexpr,
    N_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = m_offsets[:, None] * N_ + n_offsets[None, :]
    mask = (m_offsets[:, None] < M_) & (n_offsets[None, :] < N_)

    mm_values = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    addmm_values = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid = 1.0 / (tl.exp(-addmm_values) + 1.0)
    values = mm_values * sigmoid * (addmm_values * (1.0 - sigmoid) + 1.0)

    tl.store(layout_storage_ptr + offsets, values, mask=mask)
    partial = tl.sum(tl.where(mask, values, 0.0), axis=0)
    tl.store(
        partials_ptr + n_offsets * num_m_blocks + pid_m,
        partial,
        mask=n_offsets < N_,
    )


@triton.jit
def _finish_partials_kernel(
    partials_ptr,
    sum_out_ptr,
    num_m_blocks: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    feature = tl.program_id(0)
    row_blocks = tl.arange(0, BLOCK_R)
    mask = row_blocks < num_m_blocks
    values = tl.load(
        partials_ptr + feature * num_m_blocks + row_blocks,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(sum_out_ptr + feature, tl.sum(values, axis=0))


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


def _num_m_blocks(block_m: int) -> int:
    return triton.cdiv(M, block_m)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _make_workspace(
    device: torch.device,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_m_blocks = _num_m_blocks(block_m)
    layout_out = torch.empty_strided((N, M), (1, N), device=device, dtype=torch.float32)
    partials = torch.empty((N, num_m_blocks), device=device, dtype=torch.float32)
    sum_out = torch.empty((N,), device=device, dtype=torch.float32)
    return layout_out, partials, sum_out


def _oracle_into(
    mm_66: torch.Tensor,
    addmm_2: torch.Tensor,
    layout_out: torch.Tensor,
    partials: torch.Tensor,
    sum_out: torch.Tensor,
    block_m: int,
    block_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mm_66.shape == (M, N)
    assert addmm_2.shape == (M, N)
    assert mm_66.dtype is torch.float32
    assert addmm_2.dtype is torch.float32
    assert mm_66.is_contiguous()
    assert addmm_2.is_contiguous()
    assert layout_out.shape == (N, M)
    assert layout_out.stride() == (1, N)
    assert partials.shape == (N, _num_m_blocks(block_m))
    assert sum_out.shape == (N,)

    num_m_blocks = _num_m_blocks(block_m)
    _pointwise_copy_reduce_partials_kernel[(num_m_blocks, triton.cdiv(N, block_n))](
        mm_66,
        addmm_2,
        layout_out,
        partials,
        num_m_blocks=num_m_blocks,
        M_=M,
        N_=N,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    _finish_partials_kernel[(N,)](
        partials,
        sum_out,
        num_m_blocks=num_m_blocks,
        BLOCK_R=triton.next_power_of_2(num_m_blocks),
        num_warps=8,
    )
    return layout_out, sum_out


def oracle_fused(
    mm_66: torch.Tensor,
    addmm_2: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    *,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
) -> tuple[torch.Tensor, torch.Tensor]:
    layout_out, partials, sum_out = _make_workspace(mm_66.device, block_m)
    return _oracle_into(mm_66, addmm_2, layout_out, partials, sum_out, block_m, block_n)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float, block_m: int, block_n: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_outputs(inputs)
        actual = oracle_fused(*inputs, block_m=block_m, block_n=block_n)
        torch.cuda.synchronize()

    ok = True
    if len(actual) != len(ref):
        print(f"output_count_match=False actual={len(actual)} expected={len(ref)}")
        print("Correctness: FAIL")
        return False

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


def _capture_cuda_graph(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def run_bench(
    rep: int,
    warmup: int,
    no_compile: bool,
    block_m: int,
    block_n: int,
) -> dict[str, float | bool | str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    mm_66, addmm_2 = inputs[0], inputs[1]
    layout_out, partials, sum_out = _make_workspace(mm_66.device, block_m)

    timings: dict[str, float | bool | str] = {}
    with torch.no_grad():
        _oracle_into(mm_66, addmm_2, layout_out, partials, sum_out, block_m, block_n)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: _oracle_into(mm_66, addmm_2, layout_out, partials, sum_out, block_m, block_n),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    timings["oracle_us"] = oracle_us
    print(
        "oracle_fused full-scope SiLU-gradient layout+sum: "
        f"{oracle_us:.3f} us (block_m={block_m}, block_n={block_n})"
    )

    if not no_compile:
        module = _load_repro_module()
        compile_configs = [
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
        for label, config in compile_configs:
            model = module.Repro().cuda()
            with torch.no_grad():
                compiled = _compile_with_config(model, inputs, config)
                graph = _capture_cuda_graph(lambda: compiled(*inputs))
                compiled_us = triton.testing.do_bench(
                    lambda: graph.replay(),
                    warmup=warmup,
                    rep=rep,
                    return_mode="min",
                ) * 1000.0
            timings[label] = compiled_us
            print(f"torch.compile {label}: {compiled_us:.3f} us")

    compile_values = [
        value
        for key, value in timings.items()
        if key in {"coordinate_descent_tuning", "combo_looped_cd"} and isinstance(value, float)
    ]
    gates = compile_values + [HISTORICAL_BEST_COMPILE_US]
    true_floor = bool(gates and oracle_us < min(gates))
    timings["historical_best_compile_us"] = HISTORICAL_BEST_COMPILE_US
    timings["true_floor"] = true_floor
    timings["classification"] = "SCHEDULER_FUSION" if true_floor else "BANDWIDTH_BOUND"
    print(f"historical best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"Valid floor: {'yes' if true_floor else 'no (diagnosis-only)'}")
    print(
        json.dumps(
            {
                "repro_id": REPRO_ID,
                "oracle_us": oracle_us,
                "compile_us_coordinate_descent_tuning": timings.get("coordinate_descent_tuning"),
                "compile_us_combo_looped_cd": timings.get("combo_looped_cd"),
                "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
                "true_floor": true_floor,
                "classification": timings["classification"],
            },
            sort_keys=True,
        )
    )
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    if not _is_power_of_two(args.block_m) or not _is_power_of_two(args.block_n):
        raise ValueError("--block-m and --block-n must be positive powers of two")
    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(
        rtol=args.rtol,
        atol=args.atol,
        block_m=args.block_m,
        block_n=args.block_n,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            rep=args.rep,
            warmup=args.warmup,
            no_compile=args.no_compile,
            block_m=args.block_m,
            block_n=args.block_n,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
