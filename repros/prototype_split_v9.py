"""Prototype v9: Clean split with branch elimination.

Two key innovations:
1. Scheduler fix: _score_shared_output_storage() detects ConcatKernel siblings
   and gives a positive fusion score, enabling horizontal fusion into multi-store kernel.

2. Clean inner_fn construction: Instead of wrapping the original pointwise_cat inner_fn
   (which preserves dead ops.where/ops.masked branches), we construct new inner_fns that
   directly compute each half's result. This eliminates:
   - Dead masked loads (Identity-wrapped indices)
   - tl.where with always-true/false conditions
   - 4 redundant load operations per element

Results:
  - Baseline Llama form (1 kernel, pointwise_cat): 8.2 us
  - Split with wrapped inner_fn (1 kernel, dead branches): 8.2 us
  - Split with clean inner_fn (1 kernel, no branches): 6.2 us  ← TARGET HIT
  - Form B (ConcatKernel from lowering): 6.2 us

The general approach to build clean inner_fns from a pointwise_cat:
  Given: result[i] = f(x[i], rotation[i]) where rotation = cat(branch_lo, branch_hi)

  For half_lo (i ∈ [0, split)):
    rotation_lo = branch_lo(i)  (just the masked_subblock1 body)
    result_lo = f(x[i], rotation_lo)

  For half_hi (i ∈ [split, total)):
    rotation_hi = branch_hi(i)  (just the masked_subblock2 body, with offset)
    result_hi = f(x[i+split], rotation_hi)

For the specific RoPE case (x*cos + cat(-x2,x1)*sin):
  half_lo: x[d3]*cos[d3] + (-x[d3+64])*sin[d3] = x[d3]*cos[d3] - x[d3+64]*sin[d3]
  half_hi: x[d3+64]*cos[d3+64] + x[d3]*sin[d3+64]

Both halves read x at BOTH d3 and d3+64 → shared MemoryDeps → natural fusion scoring.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.virtualized import V, ops
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody
import sympy
import time
cfg.force_disable_caches = True
cfg.prefer_concat_kernel_shared_reads = False

B, S, H, D = 4, 512, 32, 128
D_HALF = D // 2


class LlamaRoPE(torch.nn.Module):
    def forward(self, x, cos, sin):
        cos_e = cos.unsqueeze(0).unsqueeze(0)
        sin_e = sin.unsqueeze(0).unsqueeze(0)
        x2 = x[..., D_HALF:]
        x1 = x[..., :D_HALF]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos_e + rotated * sin_e


x_t = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos_t = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')
sin_t = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')

from torch._inductor import scheduler as sched

original_update = GraphLowering._update_scheduler


def patched_update_scheduler(self):
    from torch._inductor.scheduler import Scheduler

    for i, op in enumerate(self.operations):
        if not isinstance(op, ir.ComputedBuffer) or not isinstance(op.data, ir.Pointwise):
            continue
        pw = op.data
        ranges = pw.ranges
        ndim = len(ranges)
        var_ranges = {}
        iter_vars = []
        for j, r in enumerate(ranges):
            sym = sympy.Symbol(f"_det{j}", integer=True, nonneg=True)
            var_ranges[sym] = r
            iter_vars.append(sym)
        try:
            body = LoopBody(pw.inner_fn, (iter_vars,), var_ranges, iter_vars, [])
        except Exception:
            continue
        masked_subblocks = [k for k in body.subblocks if k.startswith('masked_subblock')]
        if len(masked_subblocks) != 2:
            continue

        dim_idx = ndim - 1
        split_point = 64
        total_dim = 128
        orig_buf_name = op.get_name()
        original_fn = pw.inner_fn

        # Build clean inner_fns directly using ops
        def fn_lo(idx):
            i0, i1, i2, i3 = idx
            x_base = 2097152*i0 + 65536*i1 + 128*i2
            x1 = ops.load("arg2_1", x_base + i3)
            x2 = ops.load("arg2_1", x_base + i3 + 64)
            cos_val = ops.load("arg0_1", 128*i2 + i3)
            sin_val = ops.load("arg1_1", 128*i2 + i3)
            t1 = ops.mul(x1, cos_val)
            t2 = ops.mul(x2, sin_val)
            return ops.sub(t1, t2)

        def fn_hi(idx):
            i0, i1, i2, i3 = idx
            x_base = 2097152*i0 + 65536*i1 + 128*i2
            x2 = ops.load("arg2_1", x_base + i3 + 64)
            x1 = ops.load("arg2_1", x_base + i3)
            cos_val = ops.load("arg0_1", 128*i2 + i3 + 64)
            sin_val = ops.load("arg1_1", 128*i2 + i3 + 64)
            t1 = ops.mul(x1, sin_val)
            t2 = ops.mul(x2, cos_val)
            return ops.add(t1, t2)

        ranges_half = list(pw.ranges)
        ranges_half[dim_idx] = split_point

        pw_lo = ir.Pointwise(device=pw.device, dtype=pw.dtype, inner_fn=fn_lo, ranges=ranges_half)
        pw_lo._post_init_setattr("origin_node", pw.origin_node)
        pw_lo._post_init_setattr("traceback", pw.traceback)
        pw_hi = ir.Pointwise(device=pw.device, dtype=pw.dtype, inner_fn=fn_hi, ranges=ranges_half)
        pw_hi._post_init_setattr("origin_node", pw.origin_node)
        pw_hi._post_init_setattr("traceback", pw.traceback)

        sb_lo = ir.StorageBox(pw_lo)
        sb_hi = ir.StorageBox(pw_hi)
        output_size = list(pw.ranges)
        output_stride = ir.FlexibleLayout.contiguous_strides(output_size)
        concat_kernel = ir.ConcatKernel(
            name=None,
            layout=ir.FixedLayout(device=pw.device, dtype=pw.dtype, size=output_size, stride=output_stride),
            inputs=[],
        )
        kernel_storage = ir.StorageBox(concat_kernel)
        slice_lo = ir.SliceView.create(kernel_storage, dim_idx, 0, split_point, clamp=False)
        slice_hi = ir.SliceView.create(kernel_storage, dim_idx, split_point, total_dim, clamp=False)
        buf_lo = ir.ConcatKernel.realize_into(ir.TensorBox(sb_lo), slice_lo)
        buf_hi = ir.ConcatKernel.realize_into(ir.TensorBox(sb_hi), slice_hi)
        concat_kernel.inputs = ir.ConcatKernel.unwrap_storage([buf_lo, buf_hi])
        concat_kernel.name = orig_buf_name
        self.name_to_buffer[orig_buf_name] = concat_kernel
        self.register_operation(concat_kernel)
        self.operations.remove(op)
        print(f"Split {orig_buf_name} -> {buf_lo.get_name()}, {buf_hi.get_name()} (clean)")
        break

    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)


GraphLowering._update_scheduler = patched_update_scheduler

torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x_t, cos_t, sin_t)
    ref = LlamaRoPE()(x_t, cos_t, sin_t)

print(f"Correctness: max_diff={((out - ref).abs().max().item()):.6f}")


def bench_graph(fn, inputs, warmup=200, iters=1000):
    for _ in range(warmup):
        with torch.no_grad():
            fn(*inputs)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        fn(*inputs)
    torch.cuda.synchronize()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


t = bench_graph(compiled, [x_t, cos_t, sin_t])
print(f"Time: {t:.1f} us (target: 6.2 us)")

# Compare with baseline
torch._dynamo.reset()
GraphLowering._update_scheduler = original_update
compiled_baseline = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    compiled_baseline(x_t, cos_t, sin_t)
t_base = bench_graph(compiled_baseline, [x_t, cos_t, sin_t])
print(f"Baseline: {t_base:.1f} us")
print(f"Speedup: {t_base/t:.2f}x")
