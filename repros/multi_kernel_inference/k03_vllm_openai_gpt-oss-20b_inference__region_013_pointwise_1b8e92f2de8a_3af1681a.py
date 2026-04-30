"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mul_1: "f32[1, 512, 32]", getitem_24: "f32[4, 64, 512, 32]", mul_2: "f32[1, 512, 32]", getitem_25: "f32[4, 64, 512, 32]", getitem_26: "f32[4, 8, 512, 32]", getitem_27: "f32[4, 8, 512, 32]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:234 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default: "f32[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(mul_1, 1);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:228 in _apply_rotary_emb, code: first_ = first_half * cos - second_half * sin
        mul_tensor: "f32[4, 64, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_24, unsqueeze_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:235 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_1: "f32[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(mul_2, 1);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:228 in _apply_rotary_emb, code: first_ = first_half * cos - second_half * sin
        mul_tensor_1: "f32[4, 64, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_25, unsqueeze_default_1)
        sub_tensor: "f32[4, 64, 512, 32]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:229 in _apply_rotary_emb, code: second_ = second_half * cos + first_half * sin
        mul_tensor_2: "f32[4, 64, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_25, unsqueeze_default);  getitem_25 = None
        mul_tensor_3: "f32[4, 64, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_24, unsqueeze_default_1);  getitem_24 = None
        add_tensor: "f32[4, 64, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:230 in _apply_rotary_emb, code: return torch.cat((first_, second_), dim=-1)
        cat_default: "f32[4, 64, 512, 64]" = torch.ops.aten.cat.default([sub_tensor, add_tensor], -1);  sub_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:253 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        expand_default: "f32[4, 64, 512, 64]" = torch.ops.aten.expand.default(cat_default, [4, 64, 512, 64]);  cat_default = None
        reshape_default: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(expand_default, [256, 512, 64]);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:228 in _apply_rotary_emb, code: first_ = first_half * cos - second_half * sin
        mul_tensor_4: "f32[4, 8, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_26, unsqueeze_default)
        mul_tensor_5: "f32[4, 8, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_27, unsqueeze_default_1)
        sub_tensor_1: "f32[4, 8, 512, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:229 in _apply_rotary_emb, code: second_ = second_half * cos + first_half * sin
        mul_tensor_6: "f32[4, 8, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_27, unsqueeze_default);  getitem_27 = unsqueeze_default = None
        mul_tensor_7: "f32[4, 8, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_26, unsqueeze_default_1);  getitem_26 = unsqueeze_default_1 = None
        add_tensor_1: "f32[4, 8, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:230 in _apply_rotary_emb, code: return torch.cat((first_, second_), dim=-1)
        cat_default_1: "f32[4, 8, 512, 64]" = torch.ops.aten.cat.default([sub_tensor_1, add_tensor_1], -1);  sub_tensor_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:218 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_2: "f32[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(cat_default_1, 2);  cat_default_1 = None
        expand_default_1: "f32[4, 8, 8, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_default_2, [4, 8, 8, 512, 64]);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:219 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default: "f32[4, 8, 8, 512, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_1: "f32[4, 64, 512, 64]" = torch.ops.aten.reshape.default(clone_default, [4, 64, 512, 64]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:253 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        permute_default: "f32[4, 64, 64, 512]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None
        expand_default_2: "f32[4, 64, 64, 512]" = torch.ops.aten.expand.default(permute_default, [4, 64, 64, 512]);  permute_default = None
        reshape_default_2: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(expand_default_2, [256, 64, 512]);  expand_default_2 = None
        return (reshape_default, reshape_default_2)



def make_inputs():
    return [
    torch.randn([1, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 64, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 64, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 8, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 8, 512, 32], dtype=torch.float32, device='cuda'),
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
