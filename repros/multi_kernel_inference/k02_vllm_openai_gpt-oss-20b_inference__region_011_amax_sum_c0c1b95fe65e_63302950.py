"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=amax, ranges=['4', '64', '512', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['4', '64', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, cat_11: "f32[4, 64, 512, 513]", getitem_28: "f32[4, 64, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:264 in eager_attention_forward, code: combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        sub_tensor: "f32[4, 64, 512, 513]" = torch.ops.aten.sub.Tensor(cat_11, getitem_28);  cat_11 = getitem_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:265 in eager_attention_forward, code: probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        amax_default: "f32[4, 64, 512, 1]" = torch.ops.aten.amax.default(sub_tensor, [-1], True)
        sub_tensor_1: "f32[4, 64, 512, 513]" = torch.ops.aten.sub.Tensor(sub_tensor, amax_default);  sub_tensor = amax_default = None
        exp_default: "f32[4, 64, 512, 513]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[4, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[4, 64, 512, 513]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:266 in eager_attention_forward, code: scores = probs[..., :-1]  # we drop the sink here
        slice_tensor: "f32[4, 64, 512, 512]" = torch.ops.aten.slice.Tensor(div_tensor, 3, 0, -1);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:267 in eager_attention_forward, code: attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
        clone_default: "f32[4, 64, 512, 512]" = torch.ops.aten.clone.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:268 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(clone_default, [4, 64, 512, 512]);  clone_default = None
        reshape_default: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_default, [256, 512, 512]);  expand_default = None
        return reshape_default



def make_inputs():
    return [
    torch.randn([4, 64, 512, 513], dtype=torch.float32, device='cuda'),
    torch.randn([4, 64, 512, 1], dtype=torch.float32, device='cuda'),
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
