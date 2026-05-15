"""Prototype v2: Split pointwise_cat at the operations level (before Scheduler).

Instead of hacking SchedulerNodes, we transform the IR operations list.
The original ComputedBuffer with pointwise_cat is replaced by:
  - A ConcatKernel output buffer
  - Two ComputedBuffers writing to slices of that output

This way the Scheduler creates proper SchedulerNodes with all attributes.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.virtualized import V
from torch._inductor.graph import GraphLowering
import sympy
import time
import functools
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

x = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')
sin = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')


def detect_pointwise_cat_in_inner_fn(pw):
    """Detect pointwise_cat pattern by tracing the inner_fn.
    
    Returns (split_dim, split_point, total_dim) if found.
    """
    from torch._inductor.loop_body import LoopBody
    
    # Trace the inner_fn to get a LoopBody
    ranges = pw.ranges
    ndim = len(ranges)
    
    var_ranges = {}
    iter_vars = []
    for i, r in enumerate(ranges):
        sym = sympy.Symbol(f"p{i}", integer=True, nonneg=True)
        var_ranges[sym] = r
        iter_vars.append(sym)
    
    body = LoopBody(pw.inner_fn, (iter_vars,), var_ranges, iter_vars, [])
    
    # Check for masked subblocks (pointwise_cat signature)
    masked_subblocks = [k for k in body.subblocks if k.startswith('masked_subblock')]
    if len(masked_subblocks) != 2:
        return None
    
    # Find where node
    root_graph = body.root_block.graph
    where_nodes = [n for n in root_graph.nodes if n.target == 'where']
    if len(where_nodes) != 1:
        return None
    
    where_node = where_nodes[0]
    cond_node = where_node.args[1]  # lt node
    
    if cond_node.target != 'lt':
        return None
    
    # lt(ops, index_expr_result, constant_result)
    idx_expr_node = cond_node.args[1]
    const_node = cond_node.args[2]
    
    if idx_expr_node.target != 'index_expr' or const_node.target != 'constant':
        return None
    
    split_value = const_node.args[1]
    
    # Find which index key the expr references
    # The index_expr's first non-ops arg should be a get_index call
    # Actually, index_expr_node.args = (ops, get_index_result, dtype)
    # get_index_result comes from a call_module to get_index
    get_idx_node = idx_expr_node.args[1]
    if get_idx_node.op != 'call_module' or get_idx_node.target != 'get_index':
        return None
    
    idx_key = get_idx_node.args[0]
    idx_expr = body.indexing_exprs.get(idx_key)
    if idx_expr is None:
        return None
    
    # Check this is the innermost variable
    innermost_var = iter_vars[-1]
    if idx_expr != innermost_var:
        return None
    
    innermost_size = ranges[-1]
    if hasattr(innermost_size, '__int__'):
        innermost_size = int(innermost_size)
    if hasattr(split_value, '__int__'):
        split_value = int(split_value)
    
    return ndim - 1, split_value, innermost_size


def split_operations(operations):
    """Transform operations list: split pointwise_cat nodes into ConcatKernel pairs."""
    new_operations = []
    changed = False
    
    for op in operations:
        if not isinstance(op, ir.ComputedBuffer):
            new_operations.append(op)
            continue
        
        pw = op.data
        if not isinstance(pw, ir.Pointwise):
            new_operations.append(op)
            continue
        
        result = detect_pointwise_cat_in_inner_fn(pw)
        if result is None:
            new_operations.append(op)
            continue
        
        dim_idx, split_point, total_dim = result
        second_half = total_dim - split_point
        
        print(f"  Splitting {op.get_name() or 'unnamed'}: dim={dim_idx}, "
              f"split={split_point}/{total_dim}")
        
        # Create output ConcatKernel buffer
        output_size = list(pw.ranges)
        output_stride = ir.FlexibleLayout.contiguous_strides(output_size)
        
        concat_kernel = ir.ConcatKernel(
            name=None,
            layout=ir.FixedLayout(
                device=pw.device,
                dtype=pw.dtype,
                size=output_size,
                stride=output_stride,
            ),
            inputs=[],
        )
        concat_storage = ir.StorageBox(concat_kernel)
        
        # Create slice views for each half
        slice_lo = ir.SliceView.create(concat_storage, dim_idx, 0, split_point, clamp=False)
        slice_hi = ir.SliceView.create(concat_storage, dim_idx, split_point, total_dim, clamp=False)
        
        # Create inner_fns for each half
        def make_half_fn(orig_fn, offset, dim):
            def half_fn(idx):
                new_idx = list(idx)
                new_idx[dim] = idx[dim] + offset
                return orig_fn(new_idx)
            return half_fn
        
        fn_lo = make_half_fn(pw.inner_fn, 0, dim_idx)
        fn_hi = make_half_fn(pw.inner_fn, split_point, dim_idx)
        
        # Create Pointwise for each half
        ranges_lo = list(pw.ranges)
        ranges_lo[dim_idx] = split_point
        ranges_hi = list(pw.ranges)
        ranges_hi[dim_idx] = second_half
        
        pw_lo = ir.Pointwise.create(
            device=pw.device,
            dtype=pw.dtype,
            inner_fn=fn_lo,
            ranges=ranges_lo,
            origin_node=pw.origin_node,
        )
        
        pw_hi = ir.Pointwise.create(
            device=pw.device,
            dtype=pw.dtype,
            inner_fn=fn_hi,
            ranges=ranges_hi,
            origin_node=pw.origin_node,
        )
        
        # Now realize each half into the concat slices
        buf_lo = ir.ConcatKernel.realize_into(pw_lo, slice_lo)
        buf_hi = ir.ConcatKernel.realize_into(pw_hi, slice_hi)
        
        concat_kernel.inputs = ir.ConcatKernel.unwrap_storage([buf_lo, buf_hi])
        concat_kernel.name = V.graph.register_buffer(concat_kernel)
        V.graph.register_operation(concat_kernel)
        
        # The original op wrote to buf0. We need the concat to replace that.
        # Actually, the original op's buffer name is what downstream ops reference.
        # We need to make the concat kernel's output take the original name.
        
        changed = True
        # Don't add original op — it's replaced by the concat pieces
        # The realize_into calls already registered the sub-buffers as operations
        
    if changed:
        # Rebuild operations from graph
        print(f"  Split complete. Operations: {len(V.graph.operations)}")
    
    return changed


# Hook into _update_scheduler
original_update = GraphLowering._update_scheduler

def patched_update_scheduler(self):
    from .scheduler import Scheduler
    print(f"\nBefore split: {len(self.operations)} operations")
    for i, op in enumerate(self.operations):
        print(f"  [{i}] {type(op).__name__}: {op.get_name()}, "
              f"ranges={getattr(getattr(op, 'data', None), 'ranges', 'N/A')}")
    
    # Try to split
    # Note: split_operations is complex, let's just check detection for now
    for op in self.operations:
        if isinstance(op, ir.ComputedBuffer) and isinstance(op.data, ir.Pointwise):
            result = detect_pointwise_cat_in_inner_fn(op.data)
            if result:
                print(f"  DETECTED split opportunity in {op.get_name()}: {result}")
    
    original_update(self)

GraphLowering._update_scheduler = patched_update_scheduler

torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x, cos, sin)

# Verify
with torch.no_grad():
    ref = LlamaRoPE()(x, cos, sin)
print(f"\nCorrectness: max_diff={((out - ref).abs().max().item()):.6f}")
