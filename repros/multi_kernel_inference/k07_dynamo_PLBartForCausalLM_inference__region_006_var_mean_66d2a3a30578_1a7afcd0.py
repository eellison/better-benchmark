"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=var_mean, ranges=[], reduction_ranges=[]
#   origins: ['aten.var_mean.correction']
"""
import torch
import torch._inductor.config as inductor_config
from torch import device

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[50005, 768]", arg0_1: "i64[16, 1024]", arg2_1: "f32[1026, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:71 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[16, 1024, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(embedding_default, 27.712812921102035);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:979 in forward, code: cache_position = torch.arange(
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:299 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_default, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:301 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor: "i64[1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default, 2);  unsqueeze_default = None
        embedding_default_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg2_1, add_tensor);  arg2_1 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1011 in forward, code: hidden_states = inputs_embeds + positions
        add_tensor_1: "f32[16, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1);  mul_tensor = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1012 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True);  add_tensor_1 = None
        getitem: "f32[16, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:224 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.triu(causal_mask, diagonal=1)
        iota_default_1: "i64[1025]" = torch.ops.prims.iota.default(1025, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[1, 1025]" = torch.ops.aten.unsqueeze.default(iota_default_1, -2);  iota_default_1 = None
        iota_default_2: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_2: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, -1);  iota_default_2 = None
        sub_tensor: "i64[1024, 1025]" = torch.ops.aten.sub.Tensor(unsqueeze_default_1, unsqueeze_default_2);  unsqueeze_default_1 = unsqueeze_default_2 = None
        ge_scalar: "b8[1024, 1025]" = torch.ops.aten.ge.Scalar(sub_tensor, 1);  sub_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:220 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.full(
        full_default: "f32[1024, 1025]" = torch.ops.aten.full.default([1024, 1025], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:224 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.triu(causal_mask, diagonal=1)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1024, 1025]" = torch.ops.aten.where.self(ge_scalar, full_default, full_default_1);  ge_scalar = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:225 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        iota_default_3: "i64[1025]" = torch.ops.prims.iota.default(1025, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default: "i64[1024, 1]" = torch.ops.aten.reshape.default(iota_default, [-1, 1]);  iota_default = None
        gt_tensor: "b8[1024, 1025]" = torch.ops.aten.gt.Tensor(iota_default_3, reshape_default);  iota_default_3 = reshape_default = None
        mul_tensor_1: "f32[1024, 1025]" = torch.ops.aten.mul.Tensor(where_self, gt_tensor);  where_self = gt_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_3: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_4: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 1);  unsqueeze_default_3 = None
        expand_default: "f32[16, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_4, [16, 1, -1, -1]);  unsqueeze_default_4 = None
        slice_tensor: "f32[16, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default, 3, 0, 1024);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default: "f32[16, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor, [0, 8], 0.0);  slice_tensor = None
        slice_tensor_1: "f32[16, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default, -1, 0, 1024);  constant_pad_nd_default = None
        expand_default_1: "f32[16, 12, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_1, [16, 12, 1024, 1024]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_5: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_6: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 1);  unsqueeze_default_5 = None
        expand_default_2: "f32[16, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_6, [16, 1, -1, -1]);  unsqueeze_default_6 = None
        slice_tensor_2: "f32[16, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_2, 3, 0, 1024);  expand_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_1: "f32[16, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_2, [0, 8], 0.0);  slice_tensor_2 = None
        slice_tensor_3: "f32[16, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_1, -1, 0, 1024);  constant_pad_nd_default_1 = None
        expand_default_3: "f32[16, 12, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_3, [16, 12, 1024, 1024]);  slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_7: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_8: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 1);  unsqueeze_default_7 = None
        expand_default_4: "f32[16, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_8, [16, 1, -1, -1]);  unsqueeze_default_8 = None
        slice_tensor_4: "f32[16, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_4, 3, 0, 1024);  expand_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_2: "f32[16, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_4, [0, 8], 0.0);  slice_tensor_4 = None
        slice_tensor_5: "f32[16, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_2, -1, 0, 1024);  constant_pad_nd_default_2 = None
        expand_default_5: "f32[16, 12, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_5, [16, 12, 1024, 1024]);  slice_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_9: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_10: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 1);  unsqueeze_default_9 = None
        expand_default_6: "f32[16, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_10, [16, 1, -1, -1]);  unsqueeze_default_10 = None
        slice_tensor_6: "f32[16, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_6, 3, 0, 1024);  expand_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_3: "f32[16, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_6, [0, 8], 0.0);  slice_tensor_6 = None
        slice_tensor_7: "f32[16, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_3, -1, 0, 1024);  constant_pad_nd_default_3 = None
        expand_default_7: "f32[16, 12, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_7, [16, 12, 1024, 1024]);  slice_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_11: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0)
        unsqueeze_default_12: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 1);  unsqueeze_default_11 = None
        expand_default_8: "f32[16, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_12, [16, 1, -1, -1]);  unsqueeze_default_12 = None
        slice_tensor_8: "f32[16, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_8, 3, 0, 1024);  expand_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_4: "f32[16, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_8, [0, 8], 0.0);  slice_tensor_8 = None
        slice_tensor_9: "f32[16, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_4, -1, 0, 1024);  constant_pad_nd_default_4 = None
        expand_default_9: "f32[16, 12, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_9, [16, 12, 1024, 1024]);  slice_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:73 in sdpa_attention_forward, code: attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        unsqueeze_default_13: "f32[1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_14: "f32[1, 1, 1024, 1025]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 1);  unsqueeze_default_13 = None
        expand_default_10: "f32[16, 1, 1024, 1025]" = torch.ops.aten.expand.default(unsqueeze_default_14, [16, 1, -1, -1]);  unsqueeze_default_14 = None
        slice_tensor_10: "f32[16, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(expand_default_10, 3, 0, 1024);  expand_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        constant_pad_nd_default_5: "f32[16, 1, 1024, 1032]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_10, [0, 8], 0.0);  slice_tensor_10 = None
        slice_tensor_11: "f32[16, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default_5, -1, 0, 1024);  constant_pad_nd_default_5 = None
        expand_default_11: "f32[16, 12, 1024, 1024]" = torch.ops.aten.expand.default(slice_tensor_11, [16, 12, 1024, 1024]);  slice_tensor_11 = None
        return (expand_default_1, expand_default_3, expand_default_5, expand_default_7, expand_default_9, expand_default_11, getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([50005, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 100, [16, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([1026, 768], dtype=torch.float32, device='cuda'),
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
