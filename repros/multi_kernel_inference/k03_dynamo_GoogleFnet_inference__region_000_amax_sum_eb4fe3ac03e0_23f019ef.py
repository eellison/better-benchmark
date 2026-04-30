"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=amax, ranges=['16384', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['16384', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=[], reduction_ranges=[]
#   origins: ['aten.sum.default']
#   type=sum, ranges=[], reduction_ranges=[]
#   origins: ['aten.sum.default']
"""
import torch
import torch._inductor.config as inductor_config
from torch import device

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, arg113_1: "i64[32, 512]", addmm_27: "f32[16384, 32000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:685 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default: "i64[16384]" = torch.ops.aten.reshape.default(arg113_1, [-1]);  arg113_1 = None
        ne_scalar: "b8[16384]" = torch.ops.aten.ne.Scalar(reshape_default, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:340 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default_1: "f32[32, 512, 32000]" = torch.ops.aten.reshape.default(addmm_27, [32, 512, 32000]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:685 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default_2: "f32[16384, 32000]" = torch.ops.aten.reshape.default(reshape_default_1, [-1, 32000]);  reshape_default_1 = None
        amax_default: "f32[16384, 1]" = torch.ops.aten.amax.default(reshape_default_2, [1], True)
        sub_tensor: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(reshape_default_2, amax_default);  reshape_default_2 = amax_default = None
        exp_default: "f32[16384, 32000]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[16384, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar_1: "b8[16384]" = torch.ops.aten.ne.Scalar(reshape_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[16384]" = torch.ops.aten.where.self(ne_scalar_1, reshape_default, full_default);  ne_scalar_1 = full_default = None
        unsqueeze_default: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[16384, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[16384]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[16384]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[16384]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  ne_scalar = neg_default = full_default_1 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        ne_scalar_2: "b8[16384]" = torch.ops.aten.ne.Scalar(reshape_default, -100);  reshape_default = None
        sum_default_1: "i64[]" = torch.ops.aten.sum.default(ne_scalar_2);  ne_scalar_2 = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_1, torch.float32);  sum_default_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default, convert_element_type_default);  sum_default = convert_element_type_default = None
        return div_tensor



def make_inputs():
    return [
    torch.randint(0, 100, [32, 512], dtype=torch.int64, device='cuda'),
    torch.randn([16384, 32000], dtype=torch.float32, device='cuda'),
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
