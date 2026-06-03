"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full MT5 attention-softmax/dropout return from repro.py by adding the broadcast where bias to the viewed BMM tensor, doing the last-dimension max/exp/sum normalization in one persistent row kernel, applying the dropout keep mask and scale, and returning the required [192, 128, 128] transpose view, whereas Inductor currently lowers the decomposed view/add/amax/sub/exp/sum/div/inductor_random/gt/mul/expand/view/permute graph as generic reductions, pointwise RNG/dropout, and layout work with materialized softmax intermediates; Inductor cannot do this today because its scheduler and pattern matcher do not canonicalize additive-bias attention softmax with stochastic dropout and a trailing layout-only transpose into one persistent online-softmax template; the fix is NEW_PATTERN: add an Inductor lowering for last-dimension additive-bias attention softmax that fuses dropout and the output-layout epilogue into the persistent softmax kernel."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl


REPRO_ID = "amax_sum_8764401930db"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
N_HEADS = 6
Q_LEN = 128
K_LEN = 128
BH = BATCH * N_HEADS
N_ROWS = BATCH * N_HEADS * Q_LEN
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
SEED_INDEX = 37

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


@triton.jit
def _bias_softmax_dropout_kernel(
    bmm_ptr,
    where_ptr,
    random_ptr,
    out_base_ptr,
    H: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DROPOUT_P_CONST: tl.constexpr,
    DROPOUT_SCALE_CONST: tl.constexpr,
):
    row = tl.program_id(0)
    bh = row // Q
    batch = bh // H
    q = row - bh * Q

    cols = tl.arange(0, BLOCK_K)
    col_mask = cols < K
    row_offsets = row * K + cols
    where_offsets = (batch * Q + q) * K + cols

    bmm = tl.load(bmm_ptr + row_offsets, mask=col_mask, other=-float("inf")).to(tl.float32)
    bias = tl.load(where_ptr + where_offsets, mask=col_mask, other=0.0).to(tl.float32)
    x = bmm + bias

    row_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - row_max)
    denom = tl.sum(exp_x, axis=0)
    softmax = exp_x / denom

    random = tl.load(random_ptr + row_offsets, mask=col_mask, other=0.0).to(tl.float32)
    keep = random > DROPOUT_P_CONST
    out = tl.where(keep, softmax * DROPOUT_SCALE_CONST, 0.0)

    tl.store(out_base_ptr + row_offsets, out, mask=col_mask)


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
    return tuple(module.make_inputs())


def _check_inputs(
    bmm_18: torch.Tensor,
    where: torch.Tensor,
    random_values: torch.Tensor,
    out_base: torch.Tensor,
) -> None:
    assert bmm_18.is_cuda and where.is_cuda and random_values.is_cuda and out_base.is_cuda
    assert bmm_18.dtype == torch.float32
    assert where.dtype == torch.float32
    assert random_values.dtype == torch.float32
    assert out_base.dtype == torch.float32
    assert bmm_18.shape == (BH, Q_LEN, K_LEN), bmm_18.shape
    assert where.shape == (BATCH, 1, Q_LEN, K_LEN), where.shape
    assert random_values.shape == (BATCH, N_HEADS, Q_LEN, K_LEN), random_values.shape
    assert out_base.shape == (BH, Q_LEN, K_LEN), out_base.shape
    assert bmm_18.is_contiguous()
    assert where.is_contiguous()
    assert random_values.is_contiguous()
    assert out_base.is_contiguous()


def _launch_oracle(
    bmm_18: torch.Tensor,
    where: torch.Tensor,
    random_values: torch.Tensor,
    out_base: torch.Tensor,
    block_k: int = K_LEN,
) -> torch.Tensor:
    _check_inputs(bmm_18, where, random_values, out_base)
    assert block_k >= K_LEN and (block_k & (block_k - 1)) == 0

    _bias_softmax_dropout_kernel[(N_ROWS,)](
        bmm_18,
        where,
        random_values,
        out_base,
        H=N_HEADS,
        Q=Q_LEN,
        K=K_LEN,
        BLOCK_K=block_k,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
    )
    return out_base.permute(0, 2, 1)


def _inductor_random_like_repro(
    inductor_seeds: torch.Tensor,
) -> torch.Tensor:
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
    return torch.ops.prims.inductor_random.default(
        [BATCH, N_HEADS, Q_LEN, K_LEN],
        seed,
        "rand",
    )


def oracle_full_attention_softmax_dropout(
    bmm_18: torch.Tensor,
    where: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    *,
    out_base: torch.Tensor | None = None,
    random_values: torch.Tensor | None = None,
    block_k: int = K_LEN,
) -> torch.Tensor:
    assert list(_shape_param_0) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_1) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_2) == [BH, Q_LEN, K_LEN]

    if random_values is None:
        random_values = _inductor_random_like_repro(inductor_seeds)
    if out_base is None:
        out_base = torch.empty_like(bmm_18)
    return _launch_oracle(
        bmm_18,
        where,
        random_values,
        out_base,
        block_k=block_k,
    )


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / expected.float().abs().clamp_min(1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float, block_k: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    module = _load_repro_module()
    inputs = tuple(module.make_inputs())
    model = module.Repro().cuda()

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()

    with torch.no_grad():
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        expected = _as_tuple(model(*inputs))
        torch.cuda.synchronize()

        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        actual = _as_tuple(
            oracle_full_attention_softmax_dropout(*inputs, block_k=block_k)
        )
        torch.cuda.synchronize()

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        output_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={list(got.stride())} "
            f"ref_stride={list(ref.stride())} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={output_ok}"
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


def _compile_with_config(
    module,
    inputs: tuple[object, ...],
    config: dict[str, object],
    warmup: int,
):
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


def run_bench(warmup: int, rep: int, block_k: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(1)
    module = _load_repro_module()
    inputs = tuple(module.make_inputs())
    bmm_18, where, inductor_seeds, *_shape_params = inputs
    random_values = _inductor_random_like_repro(inductor_seeds)
    out_base = torch.empty_like(bmm_18)

    elems = N_ROWS * K_LEN
    dense_bytes = elems * 4 * 4
    print(
        f"oracle shape: bmm=f32[{BH}, {Q_LEN}, {K_LEN}], "
        f"where=f32[{BATCH}, 1, {Q_LEN}, {K_LEN}]"
    )
    print(f"logical fused read+write traffic: {dense_bytes / 1e6:.1f} MB")

    holder: list[torch.Tensor | None] = [None]
    with torch.no_grad():
        kernel_us = _bench_cuda(
            lambda: _launch_oracle(
                bmm_18,
                where,
                random_values,
                out_base,
                block_k=block_k,
            ),
            warmup=warmup,
            rep=rep,
        )
        full_us = _bench_cuda(
            lambda: holder.__setitem__(
                0,
                oracle_full_attention_softmax_dropout(*inputs, block_k=block_k),
            ),
            warmup=warmup,
            rep=rep,
        )

    kernel_bw = dense_bytes / (kernel_us * 1e-6) / 1e12
    full_bw = dense_bytes / (full_us * 1e-6) / 1e12
    print(
        f"oracle fused full return, precomputed random: {kernel_us:.3f} us "
        f"({kernel_bw:.3f} TB/s logical bytes)"
    )
    print(
        f"oracle full return including eager inductor_random: {full_us:.3f} us "
        f"({full_bw:.3f} TB/s logical bytes)"
    )

    if no_compile:
        return

    print("torch.compile full repro timings include Inductor RNG dropout and final permute")
    compiled_holder: list[torch.Tensor | None] = [None]
    for label, config in COMPILE_CONFIGS:
        try:
            compiled = _compile_with_config(module, inputs, config, warmup)
            us = _bench_cuda(
                lambda: compiled_holder.__setitem__(0, compiled(*inputs)),
                warmup=warmup,
                rep=rep,
            )
            print(f"torch.compile {label}: {us:.3f} us")
        except Exception as exc:
            print(f"torch.compile {label}: FAILED ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=25, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="benchmark repetitions")
    parser.add_argument("--block-k", type=int, default=K_LEN, help="Triton reduction tile size")
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

    block_k = max(args.block_k, triton.next_power_of_2(K_LEN))
    if args.check and not run_check(rtol=args.rtol, atol=args.atol, block_k=block_k):
        sys.exit(1)
    if args.bench:
        run_bench(
            warmup=args.warmup,
            rep=args.rep,
            block_k=block_k,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
