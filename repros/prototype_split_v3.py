"""Prototype v3: Split pointwise_cat at operations level.

The cleanest approach: transform IR operations before Scheduler sees them.
Replace a single ComputedBuffer(Pointwise[D]) with two ComputedBuffer(Pointwise[D_HALF])
writing to ConcatKernel slices.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.virtualized import V
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

x = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')
sin = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')


def detect_pointwise_cat_in_inner_fn(pw):
    """Detect pointwise_cat by tracing inner_fn into a LoopBody."""
    ranges = pw.ranges
    ndim = len(ranges)
    
    var_ranges = {}
    iter_vars = []
    for i, r in enumerate(ranges):
        sym = sympy.Symbol(f"_p{i}", integer=True, nonneg=True)
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
    
    where_node = where_nodes[0]
    cond_node = where_node.args[1]
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
    
    innermost_size = int(ranges[-1]) if hasattr(ranges[-1], '__int__') else None
    if innermost_size is None:
        return None
    
    return ndim - 1, int(split_value), innermost_size


def try_split_pointwise_cat_op(graph, op):
    """Try to split a ComputedBuffer with pointwise_cat into ConcatKernel pieces.
    
    Returns True if split was performed, False otherwise.
    """
    pw = op.data
    result = detect_pointwise_cat_in_inner_fn(pw)
    if result is None:
        return False
    
    dim_idx, split_point, total_dim = result
    second_half = total_dim - split_point
    
    # Only split on innermost dim for now, and only binary cat
    if dim_idx != len(pw.ranges) - 1:
        return False
    
    print(f"  Splitting: dim={dim_idx}, split={split_point}/{total_dim}")
    
    # Build half inner_fns
    # For low half: call inner_fn with idx as-is (i3 ranges [0, split_point))
    # For high half: call inner_fn with idx[dim] + split_point
    original_fn = pw.inner_fn
    
    def fn_lo(idx):
        return original_fn(idx)
    
    def fn_hi(idx):
        new_idx = list(idx)
        new_idx[dim_idx] = idx[dim_idx] + split_point
        return original_fn(new_idx)
    
    # Build ranges
    ranges_lo = list(pw.ranges)
    ranges_lo[dim_idx] = split_point
    ranges_hi = list(pw.ranges)
    ranges_hi[dim_idx] = second_half
    
    # Create Pointwise nodes (unrealized)
    pw_lo = ir.Pointwise(
        device=pw.device,
        dtype=pw.dtype,
        inner_fn=fn_lo,
        ranges=ranges_lo,
    )
    pw_lo._post_init_setattr("origin_node", pw.origin_node)
    pw_lo._post_init_setattr("traceback", pw.traceback)
    
    pw_hi = ir.Pointwise(
        device=pw.device,
        dtype=pw.dtype,
        inner_fn=fn_hi,
        ranges=ranges_hi,
    )
    pw_hi._post_init_setattr("origin_node", pw.origin_node)
    pw_hi._post_init_setattr("traceback", pw.traceback)
    
    # Wrap in TensorBox for ConcatKernel
    tb_lo = ir.TensorBox.create(pw_lo)
    tb_hi = ir.TensorBox.create(pw_hi)
    
    # Use ConcatKernel.create which handles everything
    # But we need V.graph to be set... it should be since we're in codegen
    concat_result = ir.ConcatKernel.create([tb_lo, tb_hi], dim_idx)
    
    # The original op (ComputedBuffer) wrote to a specific buffer name.
    # The ConcatKernel.create registered new operations. We need to make
    # the concat output replace the original buffer.
    
    # Get the name the concat was registered with
    # ConcatKernel.create returns a StorageBox wrapping the ConcatKernel
    # The ConcatKernel itself was registered via V.graph.register_buffer
    
    return True


# Patch _update_scheduler to run the split before scheduling
original_update = GraphLowering._update_scheduler

def patched_update_scheduler(self):
    from torch._inductor.scheduler import Scheduler
    
    split_indices = []
    for i, op in enumerate(self.operations):
        if isinstance(op, ir.ComputedBuffer) and isinstance(op.data, ir.Pointwise):
            result = detect_pointwise_cat_in_inner_fn(op.data)
            if result:
                split_indices.append(i)
                print(f"  DETECTED split in op {i} ({op.get_name()}): {result}")
    
    if split_indices:
        # For now, just do the transformation for the first detected op
        idx = split_indices[0]
        op = self.operations[idx]
        pw = op.data
        dim_idx, split_point, total_dim = detect_pointwise_cat_in_inner_fn(pw)
        second_half = total_dim - split_point
        
        original_fn = pw.inner_fn
        
        def fn_lo(idx):
            return original_fn(idx)
        
        def fn_hi(idx):
            new_idx = list(idx)
            new_idx[dim_idx] = idx[dim_idx] + split_point
            return original_fn(new_idx)
        
        ranges_lo = list(pw.ranges)
        ranges_lo[dim_idx] = split_point
        ranges_hi = list(pw.ranges)
        ranges_hi[dim_idx] = second_half
        
        pw_lo = ir.Pointwise(
            device=pw.device,
            dtype=pw.dtype,
            inner_fn=fn_lo,
            ranges=ranges_lo,
        )
        pw_lo._post_init_setattr("origin_node", pw.origin_node)
        pw_lo._post_init_setattr("traceback", pw.traceback)
        
        pw_hi = ir.Pointwise(
            device=pw.device,
            dtype=pw.dtype,
            inner_fn=fn_hi,
            ranges=ranges_hi,
        )
        pw_hi._post_init_setattr("origin_node", pw.origin_node)
        pw_hi._post_init_setattr("traceback", pw.traceback)
        
        tb_lo = ir.TensorBox.create(pw_lo)
        tb_hi = ir.TensorBox.create(pw_hi)
        
        # Use ConcatKernel.create to build proper output
        concat_result = ir.ConcatKernel.create([tb_lo, tb_hi], dim_idx)
        
        # Now: the concat registered new operations and buffers.
        # The original op's output buffer name is what graph outputs reference.
        # We need the new concat output to have the same name.
        orig_name = op.get_name()
        
        # Find the ConcatKernel that was just created
        # It should be the last NopKernel-type thing in operations
        concat_op = None
        for new_op in reversed(self.operations):
            if isinstance(new_op, ir.ConcatKernel):
                concat_op = new_op
                break
        
        if concat_op is not None:
            # Rename the concat output to match original buffer name
            old_concat_name = concat_op.get_name()
            # Replace in graph output references
            if orig_name in self.graph_outputs_by_buffer:
                # Point graph output to concat buffer
                pass
            
            # Remove original op from operations
            self.operations.remove(op)
            
            # Rename concat buffer
            if old_concat_name != orig_name:
                self.name_to_buffer[orig_name] = self.name_to_buffer.pop(old_concat_name, 
                                                                          self.name_to_buffer.get(orig_name))
                concat_op.name = orig_name
            
            print(f"  Split complete! New operations: {len(self.operations)}")
            for i, new_op in enumerate(self.operations):
                print(f"    [{i}] {type(new_op).__name__}: {new_op.get_name()}")
    
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)

GraphLowering._update_scheduler = patched_update_scheduler

torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
try:
    with torch.no_grad():
        out = compiled(x, cos, sin)
    with torch.no_grad():
        ref = LlamaRoPE()(x, cos, sin)
    print(f"\nCorrectness: max_diff={((out - ref).abs().max().item()):.6f}")
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
