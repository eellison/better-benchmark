"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete T5 additive-bias attention softmax/dropout return from Repro.forward by fusing the [64,1024,1024] to [8,8,1024,1024] view, full bias add, stable last-dimension softmax, exact Inductor seed-index-59 RNG dropout, scale, expand/view, and final transposed [64,1024,1024] output view into one persistent row Triton softmax kernel, whereas Inductor currently lowers the decomposed view/add/amax/sub/exp/sum/div/inductor_random/gt/mul/expand/view/permute graph as generic reduction, pointwise RNG/dropout, and layout kernels over materialized softmax intermediates; Inductor cannot do this today because its scheduler/pattern matcher does not canonicalize additive-bias attention softmax with stochastic Inductor RNG dropout and a trailing layout-only transpose into a single persistent online-softmax template; the fix is NEW_PATTERN: add an Inductor lowering that recognizes last-dimension additive attention softmax with RNG dropout and fuses the bias, normalization, dropout scale, and output-layout epilogue into the row-softmax schedule."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl


REPRO_ID = "amax_sum_ab3238e72dc2"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_hf_t5_train_000_04883ac2"

BATCH = 8
N_HEADS = 8
Q_LEN = 1024
K_LEN = 1024
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
SEED_INDEX = 59



@triton.jit
def _bias_softmax_dropout_kernel(
    bmm_ptr,
    bias_ptr,
    random_ptr,
    out_base_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DROPOUT_P_CONST: tl.constexpr,
    DROPOUT_SCALE_CONST: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    col_mask = cols < K
    offsets = row * K + cols

    bmm = tl.load(bmm_ptr + offsets, mask=col_mask, other=-float("inf")).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    scores = bmm + bias

    row_max = tl.max(scores, axis=0)
    numer = tl.exp(scores - row_max)
    denom = tl.sum(numer, axis=0)
    softmax = numer / denom

    random = tl.load(random_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    keep = random > DROPOUT_P_CONST
    out = tl.where(keep, softmax * DROPOUT_SCALE_CONST, 0.0)

    tl.store(out_base_ptr + offsets, out, mask=col_mask)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        value.cuda()
        if isinstance(value, torch.Tensor) and value.device.type != "cuda"
        else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _check_shape_params(
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
) -> None:
    assert list(_shape_param_0) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_1) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_2) == [BH, Q_LEN, K_LEN]


def _inductor_random_like_repro(inductor_seeds: torch.Tensor) -> torch.Tensor:
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
    return torch.ops.prims.inductor_random.default(
        [BATCH, N_HEADS, Q_LEN, K_LEN],
        seed,
        "rand",
    )


def _launch_oracle(
    bmm_34: torch.Tensor,
    add_44: torch.Tensor,
    random_values: torch.Tensor,
    out_base: torch.Tensor,
    *,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    assert bmm_34.is_cuda and add_44.is_cuda and random_values.is_cuda
    assert out_base.is_cuda
    assert bmm_34.dtype == torch.float32 and bmm_34.shape == (BH, Q_LEN, K_LEN)
    assert add_44.dtype == torch.float32 and add_44.shape == (
        BATCH,
        N_HEADS,
        Q_LEN,
        K_LEN,
    )
    assert random_values.dtype == torch.float32
    assert random_values.shape == (BATCH, N_HEADS, Q_LEN, K_LEN)
    assert out_base.dtype == torch.float32 and out_base.shape == bmm_34.shape
    assert bmm_34.is_contiguous()
    assert add_44.is_contiguous()
    assert random_values.is_contiguous()
    assert out_base.is_contiguous()
    assert block_k >= K_LEN and (block_k & (block_k - 1)) == 0

    _bias_softmax_dropout_kernel[(N_ROWS,)](
        bmm_34,
        add_44,
        random_values,
        out_base,
        K=K_LEN,
        BLOCK_K=block_k,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        num_warps=num_warps,
    )
    return out_base.permute(0, 2, 1)


def oracle_full_attention_softmax_dropout(
    bmm_34: torch.Tensor,
    add_44: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    *,
    out_base: torch.Tensor | None = None,
    random_values: torch.Tensor | None = None,
    block_k: int = K_LEN,
    num_warps: int = 8,
) -> torch.Tensor:
    _check_shape_params(_shape_param_0, _shape_param_1, _shape_param_2)

    if random_values is None:
        random_values = _inductor_random_like_repro(inductor_seeds)
    if out_base is None:
        out_base = torch.empty_like(bmm_34)
    return _launch_oracle(
        bmm_34,
        add_44,
        random_values,
        out_base,
        block_k=block_k,
        num_warps=num_warps,
    )


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    finite_diff = diff[torch.isfinite(diff)]
    max_abs = finite_diff.max().item() if finite_diff.numel() else float("nan")
    rel = diff / expected.float().abs().clamp_min(1e-8)
    finite_rel = rel[torch.isfinite(rel)]
    max_rel = finite_rel.max().item() if finite_rel.numel() else float("nan")
    return max_abs, max_rel


def run_check(rtol: float, atol: float, block_k: int, num_warps: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()

    with torch.no_grad():
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        expected = reference_outputs(inputs)
        torch.cuda.synchronize()

        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        actual = _as_tuple(
            oracle_full_attention_softmax_dropout(
                *inputs,
                block_k=block_k,
                num_warps=num_warps,
            )
        )
        torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(f"output_count: actual={len(actual)} expected={len(expected)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        metadata_ok = (
            got.shape == ref.shape
            and got.dtype == ref.dtype
            and got.stride() == ref.stride()
        )
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(
            got.float(),
            ref.float(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        output_ok = metadata_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} expected_shape={list(ref.shape)} "
            f"dtype={got.dtype} expected_dtype={ref.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} metadata={metadata_ok}"
        )

    print("check compared against full Repro.forward return value, including dropout")
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


def run_bench(warmup: int, rep: int, block_k: int, num_warps: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(1)
    inputs = make_inputs()
    bmm_34, add_44, inductor_seeds, *_shape_params = inputs
    random_values = _inductor_random_like_repro(inductor_seeds)
    out_base = torch.empty_like(bmm_34)

    elems = N_ROWS * K_LEN
    kernel_bytes = elems * (4 + 4 + 4 + 4)
    full_bytes = kernel_bytes
    print(
        f"oracle shape: bmm=f32[{BH}, {Q_LEN}, {K_LEN}], "
        f"bias=f32[{BATCH}, {N_HEADS}, {Q_LEN}, {K_LEN}]"
    )
    print(f"logical fused read+write traffic: {kernel_bytes / 1e6:.1f} MB")

    holder: list[torch.Tensor | None] = [None]
    with torch.no_grad():
        kernel_us = _bench_cuda(
            lambda: _launch_oracle(
                bmm_34,
                add_44,
                random_values,
                out_base,
                block_k=block_k,
                num_warps=num_warps,
            ),
            warmup=warmup,
            rep=rep,
        )
        full_us = _bench_cuda(
            lambda: holder.__setitem__(
                0,
                oracle_full_attention_softmax_dropout(
                    *inputs,
                    block_k=block_k,
                    num_warps=num_warps,
                ),
            ),
            warmup=warmup,
            rep=rep,
        )

    kernel_bw = kernel_bytes / (kernel_us * 1e-6) / 1e12
    full_bw = full_bytes / (full_us * 1e-6) / 1e12
    print(
        f"oracle fused full return, precomputed random: {kernel_us:.3f} us "
        f"({kernel_bw:.3f} TB/s logical bytes)"
    )
    print(
        f"oracle full return including eager inductor_random: {full_us:.3f} us "
        f"({full_bw:.3f} TB/s logical bytes) shape={SHAPE_LABEL}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="benchmark repetitions")
    parser.add_argument("--block-k", type=int, default=K_LEN, help="Triton reduction tile size")
    parser.add_argument("--num-warps", type=int, default=8, help="Triton softmax warps")
    parser.add_argument("--rtol", type=float, default=5e-5)
    parser.add_argument("--atol", type=float, default=5e-6)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    block_k = max(args.block_k, triton.next_power_of_2(K_LEN))
    if args.check and not run_check(
        rtol=args.rtol,
        atol=args.atol,
        block_k=block_k,
        num_warps=args.num_warps,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            warmup=args.warmup,
            rep=args.rep,
            block_k=block_k,
            num_warps=args.num_warps,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
