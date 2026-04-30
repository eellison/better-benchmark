"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=any, ranges=[], reduction_ranges=[]
#   origins: ['aten.any.dims']
#   type=any, ranges=[], reduction_ranges=[]
#   origins: ['aten.any.dims']
"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, getitem_1: "i64[2048, 8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:246 in forward, code: expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        ge_scalar: "b8[2048, 8]" = torch.ops.aten.ge.Scalar(getitem_1, 0)
        logical_not_default: "b8[2048, 8]" = torch.ops.aten.logical_not.default(ge_scalar);  ge_scalar = None
        any_dims: "b8[]" = torch.ops.aten.any.dims(logical_not_default);  logical_not_default = None
        logical_not_default_1: "b8[]" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None
        _assert_async_msg = torch.ops.aten._assert_async.msg(logical_not_default_1, 'one_hot: Class values must be non-negative.');  logical_not_default_1 = None
        lt_scalar: "b8[2048, 8]" = torch.ops.aten.lt.Scalar(getitem_1, 128)
        logical_not_default_2: "b8[2048, 8]" = torch.ops.aten.logical_not.default(lt_scalar);  lt_scalar = None
        any_dims_1: "b8[]" = torch.ops.aten.any.dims(logical_not_default_2);  logical_not_default_2 = None
        logical_not_default_3: "b8[]" = torch.ops.aten.logical_not.default(any_dims_1);  any_dims_1 = None
        _assert_async_msg_1 = torch.ops.aten._assert_async.msg(logical_not_default_3, 'one_hot: Class values must be smaller than num_classes.');  logical_not_default_3 = None
        unsqueeze_default: "i64[2048, 8, 1]" = torch.ops.aten.unsqueeze.default(getitem_1, -1);  getitem_1 = None
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        eq_tensor: "b8[2048, 8, 128]" = torch.ops.aten.eq.Tensor(unsqueeze_default, iota_default);  unsqueeze_default = iota_default = None
        convert_element_type_default: "i64[2048, 8, 128]" = torch.ops.prims.convert_element_type.default(eq_tensor, torch.int64);  eq_tensor = None
        permute_default: "i64[128, 8, 2048]" = torch.ops.aten.permute.default(convert_element_type_default, [2, 1, 0]);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:249 in forward, code: expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        sum_dim_int_list: "i64[128]" = torch.ops.aten.sum.dim_IntList(permute_default, [-1, -2]);  permute_default = None
        gt_scalar: "b8[128]" = torch.ops.aten.gt.Scalar(sum_dim_int_list, 0);  sum_dim_int_list = None
        return (gt_scalar, _assert_async_msg_1, _assert_async_msg)



def make_inputs():
    return [
    torch.randint(0, 100, [2048, 8], dtype=torch.int64, device='cuda'),
    ]


def _count_bytes(inputs, outputs):
    """Count total read + write bytes for SOL calculation."""
    total = 0
    for t in inputs:
        if isinstance(t, torch.Tensor):
            total += t.nelement() * t.element_size()
    if isinstance(outputs, torch.Tensor):
        total += outputs.nelement() * outputs.element_size()
    elif isinstance(outputs, (tuple, list)):
        for o in outputs:
            if isinstance(o, torch.Tensor):
                total += o.nelement() * o.element_size()
    return total


def _measure_memcopy_sol(total_bytes, n_warmup=25, n_iter=200):
    """Measure memcopy speed-of-light at multiple transfer sizes.

    Returns dict mapping size label to time in microseconds.
    We measure at the exact kernel size plus a few reference sizes
    so we can see where on the bandwidth curve this kernel sits.
    """
    results = {}
    sizes = set()
    sizes.add(total_bytes)
    # Add reference sizes: 1KB, 64KB, 1MB, 16MB, 64MB, 256MB
    for ref in [1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 256*1024*1024]:
        sizes.add(ref)

    for nbytes in sorted(sizes):
        n_elems = max(nbytes // 4, 1)
        src = torch.empty(n_elems, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)
        for _ in range(n_warmup):
            dst.copy_(src)
        torch.cuda.synchronize()

        import time
        start = time.perf_counter()
        for _ in range(n_iter):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iter

        bw_gbps = (nbytes * 2) / elapsed / 1e9  # read + write
        label = f"{nbytes / 1024:.0f}KB" if nbytes < 1024*1024 else f"{nbytes / 1024 / 1024:.1f}MB"
        is_kernel_size = (nbytes == total_bytes)
        results[nbytes] = {
            "label": label,
            "time_us": elapsed * 1e6,
            "bw_gbps": bw_gbps,
            "is_kernel_size": is_kernel_size,
        }
    return results


def benchmark(n_warmup=25, n_iter=200):
    mod = Repro()
    inputs = make_inputs()

    # Eager baseline
    with torch.no_grad():
        eager_out = mod(*inputs)

    total_bytes = _count_bytes(inputs, eager_out)

    # Compiled (default heuristics)
    compiled = torch.compile(mod)
    with torch.no_grad():
        for _ in range(n_warmup):
            compiled(*inputs)
        torch.cuda.synchronize()

        import time
        start = time.perf_counter()
        for _ in range(n_iter):
            compiled(*inputs)
        torch.cuda.synchronize()
        compiled_time = (time.perf_counter() - start) / n_iter * 1e6

    # Compiled with coordinate descent tuning
    inductor_config.coordinate_descent_tuning = True
    torch._dynamo.reset()
    compiled_cd = torch.compile(mod)
    with torch.no_grad():
        for _ in range(n_warmup):
            compiled_cd(*inputs)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iter):
            compiled_cd(*inputs)
        torch.cuda.synchronize()
        cd_time = (time.perf_counter() - start) / n_iter * 1e6

    # Memcopy SOL at multiple sizes
    sol_results = _measure_memcopy_sol(total_bytes)
    kernel_sol = sol_results[total_bytes]

    print(f"\nKernel data: {total_bytes / 1024:.1f} KB (read+write)")
    print(f"Compiled (default):      {compiled_time:8.1f} us")
    print(f"Compiled (coord descent):{cd_time:8.1f} us")
    print(f"Memcopy SOL (same size): {kernel_sol['time_us']:8.1f} us  ({kernel_sol['bw_gbps']:.1f} GB/s)")
    print(f"Gap (default / SOL):     {compiled_time / kernel_sol['time_us']:8.2f}x")
    print(f"Gap (CD / SOL):          {cd_time / kernel_sol['time_us']:8.2f}x")
    print(f"\nMemcopy bandwidth curve:")
    for nbytes in sorted(sol_results):
        r = sol_results[nbytes]
        marker = " <-- kernel size" if r["is_kernel_size"] else ""
        print(f"  {r['label']:>10s}: {r['time_us']:8.1f} us  {r['bw_gbps']:8.1f} GB/s{marker}")

    return {
        "compiled_us": compiled_time,
        "coord_descent_us": cd_time,
        "memcopy_sol_us": kernel_sol["time_us"],
        "total_bytes": total_bytes,
        "sol_curve": sol_results,
    }


if __name__ == "__main__":
    benchmark()
