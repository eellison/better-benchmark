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
    def forward(self):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:772 in forward, code: attention_mask = torch.ones(input_shape, device=device)
        full_default: "f32[8, 512]" = torch.ops.aten.full.default([8, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:609 in get_attention_mask, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        unsqueeze_default: "f32[8, 1, 512]" = torch.ops.aten.unsqueeze.default(full_default, 1);  full_default = None
        unsqueeze_default_1: "f32[8, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:610 in get_attention_mask, code: attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        squeeze_dim: "f32[8, 1, 512]" = torch.ops.aten.squeeze.dim(unsqueeze_default_1, -2)
        unsqueeze_default_2: "f32[8, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(squeeze_dim, -1);  squeeze_dim = None
        mul_tensor: "f32[8, 1, 512, 512]" = torch.ops.aten.mul.Tensor(unsqueeze_default_1, unsqueeze_default_2);  unsqueeze_default_1 = unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:260 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_1: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_2: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_3: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_4: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_5: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_6: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_7: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_8: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_9: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_10: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_11: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_12: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_13: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_14: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_15: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_16: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_17: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_18: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_19: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_20: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_21: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_22: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_23: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool);  mul_tensor = None
        return (convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_7, convert_element_type_default_8, convert_element_type_default_9, convert_element_type_default_10, convert_element_type_default_11, convert_element_type_default_12, convert_element_type_default_13, convert_element_type_default_14, convert_element_type_default_15, convert_element_type_default_16, convert_element_type_default_17, convert_element_type_default_18, convert_element_type_default_19, convert_element_type_default_20, convert_element_type_default_21, convert_element_type_default_22, convert_element_type_default_23)



def make_inputs():
    return [

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
