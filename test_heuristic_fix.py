"""
Test that the outer_config_opt heuristic fix produces correct and fast kernels.

Tests various (x, rnumel) combinations to verify no regressions.
"""
import torch
import torch._inductor.config as config
import time

config.split_reductions = False
config.force_disable_caches = True


def make_outer_reduction_model(xnumel, rnumel):
    """Create an outer reduction: [rnumel, xnumel] -> [xnumel] via sum(dim=0)."""
    class Model(torch.nn.Module):
        def forward(self, a, b, c):
            return (a * b * c).sum(dim=0, keepdim=True)
    return Model()


def benchmark_kernel(xnumel, rnumel, n_warmup=50, n_iter=500):
    model = make_outer_reduction_model(xnumel, rnumel)
    a = torch.randn(rnumel, xnumel, device="cuda")
    b = torch.randn(rnumel, xnumel, device="cuda")
    c = torch.randn(rnumel, 1, device="cuda")

    compiled = torch.compile(model)
    with torch.no_grad():
        expected = model(a, b, c)
        actual = compiled(a, b, c)
        assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2), \
            f"x={xnumel}, r={rnumel}: max diff {(actual - expected).abs().max().item()}"

        for _ in range(n_warmup):
            compiled(a, b, c)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iter):
            compiled(a, b, c)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iter * 1e6

    torch._dynamo.reset()
    return elapsed


def main():
    cases = [
        (4096, 2048),
        (4096, 1024),
        (4096, 512),
        (4096, 4096),
        (8192, 2048),
        (2048, 2048),
        (1024, 2048),
        (16384, 1024),
    ]

    print(f"{'xnumel':>8} {'rnumel':>8} {'Time(us)':>10}")
    print("-" * 30)
    for x, r in cases:
        try:
            t = benchmark_kernel(x, r)
            print(f"{x:>8} {r:>8} {t:>9.1f}")
        except Exception as e:
            print(f"{x:>8} {r:>8} FAIL: {e}")


if __name__ == "__main__":
    main()
