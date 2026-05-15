"""Test persistent reduction threshold for var_mean kernels on B200."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")

import torch
import torch._inductor.config as cfg

cfg.force_disable_caches = True

def bench_var_mean(rows, rdim, label, persistent_threshold=None):
    """Benchmark var_mean with optional persistent threshold override."""
    from triton.testing import do_bench

    if persistent_threshold is not None:
        from unittest.mock import patch
        from torch._inductor import choices
        orig = choices.InductorChoices.should_use_persistent_reduction

        @staticmethod
        def patched_should_use_persistent(features, cooperative_reduction):
            from torch._inductor.ir import ReductionHint
            if not cfg.triton.persistent_reductions:
                return False
            threshold = {
                ReductionHint.INNER: 1024,
            }.get(features.get_reduction_hint(), persistent_threshold)

            from torch._inductor.codegen.simd import SIMDScheduling
            from torch._inductor.utils import bound_sympy
            from torch._inductor.runtime.runtime_utils import next_power_of_2
            from torch._inductor import virtualized as V

            if features.get_reduction_hint() not in (
                ReductionHint.INNER,
                ReductionHint.OUTER_TINY,
            ):
                bounds = bound_sympy(features.reduction_numel)
                lower = bounds.lower
                upper = bounds.upper
                if not all(
                    (
                        (isinstance(bound, int) or bound.is_constant())
                        and not torch.utils._sympy.numbers.is_infinite(bound)
                    )
                    for bound in (lower, upper)
                ):
                    return False
                lower = next_power_of_2(int(lower))
                upper = next_power_of_2(int(upper))
                if lower != upper:
                    return False

            if cooperative_reduction:
                threshold *= 32 // min(
                    V.graph.sizevars.optimization_hint(features.numel), 32
                )
            if cfg.triton.multi_kernel:
                threshold *= 16

            return V.graph.sizevars.statically_known_leq(
                features.reduction_numel, threshold
            )

        choices.InductorChoices.should_use_persistent_reduction = patched_should_use_persistent

    x = torch.randn(rows, rdim, dtype=torch.float32, device='cuda')

    def fn(x):
        reshaped = x.view(-1, rdim)
        # Simulate GELU + var_mean (the common fused pattern)
        h = torch.nn.functional.gelu(reshaped, approximate='tanh')
        return torch.var_mean(h, dim=-1, correction=0, keepdim=True)

    compiled = torch.compile(fn)
    with torch.no_grad():
        compiled(x)  # warmup
        compiled(x)  # 2nd warmup

    # CUDA graph benchmark
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            compiled(x)
    torch.cuda.current_stream().wait_stream(stream)

    # Warmup graph replay
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()

    import time
    n_iter = 200
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        g.replay()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    graph_us = (t1 - t0) / n_iter * 1e6

    # SOL
    total_bytes = rows * rdim * 4 + 2 * (rows * 4)  # input + 2 outputs (var, mean)
    sol_bytes = rows * rdim * 4 * 2  # approx: read input once for GELU, once for var_mean
    src = torch.empty(total_bytes // 4, dtype=torch.float32, device='cuda')
    dst = torch.empty_like(src)
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        dst.copy_(src)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    sol_us = (t1 - t0) / n_iter * 1e6

    print(f"  {label}: graph={graph_us:.1f}us  sol={sol_us:.1f}us  gap={graph_us/sol_us:.2f}x")
    return graph_us, sol_us


def main():
    test_cases = [
        (16384, 768, "GoogleFnet/Bert (16384x768)"),
        (4096, 1536, "DebertaV2 (4096x1536)"),
        (8192, 768, "Longformer (8192x768)"),
        (32768, 768, "DistilBert (32768x768)"),
        (32768, 256, "Reformer (32768x256)"),
        (32768, 128, "Electra (32768x128)"),
        (2048, 2560, "Blenderbot (2048x2560)"),
    ]

    print("=== Baseline (persistent threshold = 64) ===")
    baseline = {}
    for rows, rdim, label in test_cases:
        g, s = bench_var_mean(rows, rdim, label, persistent_threshold=None)
        baseline[label] = (g, s)

    print()
    print("=== With persistent threshold = 1024 ===")
    patched = {}
    for rows, rdim, label in test_cases:
        g, s = bench_var_mean(rows, rdim, label, persistent_threshold=1024)
        patched[label] = (g, s)

    print()
    print("=== Summary ===")
    for rows, rdim, label in test_cases:
        bg, bs = baseline[label]
        pg, ps = patched[label]
        speedup = bg / pg if pg > 0 else 0
        print(f"  {label}: {bg:.1f}us -> {pg:.1f}us  ({speedup:.2f}x speedup)  rdim={rdim}")


if __name__ == "__main__":
    main()
