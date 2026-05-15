"""Prototype v7: Debug WHY the two halves don't fuse.

The halves should fuse because:
1. Same numel (4194304)
2. Same device
3. Share input reads (arg0_1, arg1_1, arg2_1)
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.virtualized import V
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody
import sympy, time, logging
cfg.force_disable_caches = True
cfg.prefer_concat_kernel_shared_reads = False

# Enable fusion logging
logging.getLogger("torch._inductor.scheduler").setLevel(logging.DEBUG)

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

x = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')
sin = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')


def detect_pointwise_cat_split(pw):
    ranges = pw.ranges
    ndim = len(ranges)
    var_ranges = {}
    iter_vars = []
    for i, r in enumerate(ranges):
        sym = sympy.Symbol(f"_det{i}", integer=True, nonneg=True)
        var_ranges[sym] = r
        iter_vars.append(sym)
    try:
        body = LoopBody(pw.inner_fn, (iter_vars,), var_ranges, iter_vars, [])
    except Exception:
        return None
    masked_subblocks = [k for k in body.subblocks if k.startswith('masked_subblock')]
    if len(masked_subblocks) != 2:
        return None
    root_graph = body.root_block.graph
    where_nodes = [n for n in root_graph.nodes if n.target == 'where']
    if len(where_nodes) != 1:
        return None
    cond_node = where_nodes[0].args[1]
    if cond_node.target != 'lt':
        return None
    idx_expr_node = cond_node.args[1]
    const_node = cond_node.args[2]
    if idx_expr_node.target != 'index_expr' or const_node.target != 'constant':
        return None
    split_value = const_node.args[1]
    get_idx_node = idx_expr_node.args[1]
    if get_idx_node.op != 'call_module' or get_idx_node.target != 'get_index':
        return None
    idx_key = get_idx_node.args[0]
    idx_expr = body.indexing_exprs.get(idx_key)
    if idx_expr is None:
        return None
    innermost_var = iter_vars[-1]
    if idx_expr != innermost_var:
        return None
    try:
        return ndim - 1, int(split_value), int(ranges[-1])
    except (TypeError, ValueError):
        return None


original_update = GraphLowering._update_scheduler

def patched_update_scheduler(self):
    from torch._inductor.scheduler import Scheduler
    
    for i, op in enumerate(self.operations):
        if not isinstance(op, ir.ComputedBuffer) or not isinstance(op.data, ir.Pointwise):
            continue
        result = detect_pointwise_cat_split(op.data)
        if result is None:
            continue
        
        dim_idx, split_point, total_dim = result
        second_half = total_dim - split_point
        pw = op.data
        orig_buf_name = op.get_name()
        
        original_fn = pw.inner_fn
        def fn_lo(idx, _fn=original_fn):
            return _fn(idx)
        def fn_hi(idx, _fn=original_fn, _sp=split_point, _di=dim_idx):
            new_idx = list(idx)
            new_idx[_di] = idx[_di] + _sp
            return _fn(new_idx)
        
        ranges_lo = list(pw.ranges); ranges_lo[dim_idx] = split_point
        ranges_hi = list(pw.ranges); ranges_hi[dim_idx] = second_half
        
        pw_lo = ir.Pointwise(device=pw.device, dtype=pw.dtype, inner_fn=fn_lo, ranges=ranges_lo)
        pw_lo._post_init_setattr("origin_node", pw.origin_node)
        pw_lo._post_init_setattr("traceback", pw.traceback)
        pw_hi = ir.Pointwise(device=pw.device, dtype=pw.dtype, inner_fn=fn_hi, ranges=ranges_hi)
        pw_hi._post_init_setattr("origin_node", pw.origin_node)
        pw_hi._post_init_setattr("traceback", pw.traceback)
        
        sb_lo = ir.StorageBox(pw_lo)
        sb_hi = ir.StorageBox(pw_hi)
        
        output_size = list(pw.ranges)
        output_stride = ir.FlexibleLayout.contiguous_strides(output_size)
        concat_kernel = ir.ConcatKernel(
            name=None,
            layout=ir.FixedLayout(device=pw.device, dtype=pw.dtype,
                                  size=output_size, stride=output_stride),
            inputs=[],
        )
        kernel_storage = ir.StorageBox(concat_kernel)
        slice_lo = ir.SliceView.create(kernel_storage, dim_idx, 0, split_point, clamp=False)
        slice_hi = ir.SliceView.create(kernel_storage, dim_idx, split_point, total_dim, clamp=False)
        
        tb_lo = ir.TensorBox(sb_lo)
        tb_hi = ir.TensorBox(sb_hi)
        buf_lo = ir.ConcatKernel.realize_into(tb_lo, slice_lo)
        buf_hi = ir.ConcatKernel.realize_into(tb_hi, slice_hi)
        
        concat_kernel.inputs = ir.ConcatKernel.unwrap_storage([buf_lo, buf_hi])
        concat_kernel.name = orig_buf_name
        self.name_to_buffer[orig_buf_name] = concat_kernel
        self.register_operation(concat_kernel)
        self.operations.remove(op)
        
        print(f"  Split {orig_buf_name} -> {buf_lo.get_name()}, {buf_hi.get_name()}")
        break
    
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)

GraphLowering._update_scheduler = patched_update_scheduler

torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x, cos, sin)
