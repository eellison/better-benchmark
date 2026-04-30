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
    def forward(self, mm_46: "f32[8192, 768]", arg184_1: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:510 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_46, [1024, 8, 768]);  mm_46 = None
        add_tensor: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(reshape_default, arg184_1);  reshape_default = arg184_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:598 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        reshape_default_1: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_tensor, [1024, 8, 12, 64]);  add_tensor = None
        permute_default: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0, 2, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_default_1: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_2: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_default_1, [96, 1024, 64]);  permute_default_1 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(reshape_default_2, [0, 0, 256, 256], -1.0);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:873 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_default: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_default, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:877 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_default: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        clone_default: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default, memory_format = torch.contiguous_format);  unsqueeze_default = None
        reshape_default_3: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_default, [384, 768, 64]);  clone_default = None
        return reshape_default_3



def make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
