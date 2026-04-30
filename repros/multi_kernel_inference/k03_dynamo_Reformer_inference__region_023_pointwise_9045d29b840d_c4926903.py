"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config
from torch import device

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, bmm_2: "f32[6144, 64, 64]", getitem_1: "i64[8, 12, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:930 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        reshape_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_2, [8, 12, 64, 64, 64]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:938 in _attend, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        reshape_default_1: "f32[8, 12, 4096, 64]" = torch.ops.aten.reshape.default(reshape_default, [8, 12, 4096, 64]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:805 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        empty_memory_format: "i64[8, 12, 4096]" = torch.ops.aten.empty.memory_format([8, 12, 4096], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:799 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
        iota_default: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:800 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .view(1, 1, -1)
        reshape_default_2: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_default, [1, 1, -1]);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:801 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .expand(sorted_bucket_idx.shape)
        expand_default: "i64[8, 12, 4096]" = torch.ops.aten.expand.default(reshape_default_2, [8, 12, 4096]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:806 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)
        scatter_src: "i64[8, 12, 4096]" = torch.ops.aten.scatter.src(empty_memory_format, -1, getitem_1, expand_default);  empty_memory_format = getitem_1 = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1114 in forward, code: expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
        unsqueeze_default: "i64[8, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(scatter_src, -1);  scatter_src = None
        expand_default_1: "i64[8, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default, [8, 12, 4096, 64]);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1115 in forward, code: out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
        gather_default: "f32[8, 12, 4096, 64]" = torch.ops.aten.gather.default(reshape_default_1, 2, expand_default_1);  reshape_default_1 = expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:413 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_default: "f32[8, 4096, 12, 64]" = torch.ops.aten.permute.default(gather_default, [0, 2, 1, 3]);  gather_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:414 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_default: "f32[8, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(clone_default, [8, 4096, 768]);  clone_default = None
        return reshape_default_3



def make_inputs():
    return [
    torch.randn([6144, 64, 64], dtype=torch.float32, device='cuda'),
    torch.randint(0, 100, [8, 12, 4096], dtype=torch.int64, device='cuda'),
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
