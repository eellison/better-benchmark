"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=argmax, ranges=['8', '12', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.argmax.default']
"""
import torch
import torch._inductor.config as inductor_config
from torch import device

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, bmm: "f32[12, 32768, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:750 in _hash_vectors, code: rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
        reshape_default: "f32[12, 8, 4096, 1, 1, 64]" = torch.ops.aten.reshape.default(bmm, [12, 8, 4096, 1, 1, 64]);  bmm = None
        permute_default: "f32[8, 12, 1, 4096, 64, 1]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 4, 2, 5, 3]);  reshape_default = None
        reshape_default_1: "f32[8, 12, 1, 4096, 64]" = torch.ops.aten.reshape.default(permute_default, [8, 12, 1, 4096, 64]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:753 in _hash_vectors, code: rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
        neg_default: "f32[8, 12, 1, 4096, 64]" = torch.ops.aten.neg.default(reshape_default_1)
        cat_default: "f32[8, 12, 1, 4096, 128]" = torch.ops.aten.cat.default([reshape_default_1, neg_default], -1);  reshape_default_1 = neg_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:754 in _hash_vectors, code: buckets = torch.argmax(rotated_vectors, dim=-1)
        argmax_default: "i64[8, 12, 1, 4096]" = torch.ops.aten.argmax.default(cat_default, -1);  cat_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:782 in _hash_vectors, code: offsets = torch.arange(num_hashes, device=vectors.device)
        iota_default: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:783 in _hash_vectors, code: offsets = (offsets * num_buckets).view((1, 1, -1, 1))
        mul_tensor: "i64[1]" = torch.ops.aten.mul.Tensor(iota_default, 128);  iota_default = None
        reshape_default_2: "i64[1, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor, [1, 1, -1, 1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:786 in _hash_vectors, code: offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        expand_default: "i64[8, 12, 1, 1]" = torch.ops.aten.expand.default(reshape_default_2, [8, 12, 1, 1]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:787 in _hash_vectors, code: offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)
        add_tensor: "i64[8, 12, 1, 4096]" = torch.ops.aten.add.Tensor(argmax_default, expand_default);  argmax_default = expand_default = None
        reshape_default_3: "i64[8, 12, 4096]" = torch.ops.aten.reshape.default(add_tensor, [8, 12, 4096]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:181 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        mul_tensor_1: "i64[8, 12, 4096]" = torch.ops.aten.mul.Tensor(reshape_default_3, 4096)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:179 in _stable_argsort, code: scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
        iota_default_1: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_4: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_default_1, [1, 1, -1]);  iota_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:180 in _stable_argsort, code: scale_offset = scale_offset.expand(vector.shape)
        expand_default_1: "i64[8, 12, 4096]" = torch.ops.aten.expand.default(reshape_default_4, [8, 12, 4096]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:181 in _stable_argsort, code: scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
        remainder_scalar: "i64[8, 12, 4096]" = torch.ops.aten.remainder.Scalar(expand_default_1, 4096);  expand_default_1 = None
        add_tensor_1: "i64[8, 12, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, remainder_scalar);  mul_tensor_1 = remainder_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:182 in _stable_argsort, code: return torch.argsort(scaled_vector, dim=dim)
        sort_default = torch.ops.aten.sort.default(add_tensor_1);  add_tensor_1 = None
        getitem: "i64[8, 12, 4096]" = sort_default[1];  sort_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:703 in torch_dynamo_resume_in_forward_at_572, code: self.num_attention_heads, self.attention_head_size, self.hidden_size
        reshape_default_5: "i64[8, 12, 1, 4096]" = torch.ops.aten.reshape.default(reshape_default_3, [8, 12, 1, -1]);  reshape_default_3 = None
        return (reshape_default_5, getitem)



def make_inputs():
    return [
    torch.randn([12, 32768, 64], dtype=torch.float32, device='cuda'),
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
