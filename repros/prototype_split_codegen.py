"""Prototype: split a pointwise_cat node into two half-D nodes at scheduler level.

The approach:
1. Detect pointwise_cat pattern in SchedulerNode LoopBody
2. Create two new Pointwise IR nodes with modified inner_fns
3. Create a ConcatKernel-style output
4. Replace the original node with two new SchedulerNodes

For inner_fn construction: call the ORIGINAL inner_fn with offset indices.
Use a custom tracing that folds the branch condition (since we know the range).
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor import scheduler as sched_mod
from torch._inductor.virtualized import V
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

def detect_pointwise_cat_split(body):
    """Detect if a LoopBody has a pointwise_cat pattern on innermost dim.
    Returns (split_point, innermost_var) if found.
    """
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
    
    lt_args = cond_node.args
    idx_expr_node = lt_args[1]
    const_node = lt_args[2]
    
    if idx_expr_node.target != 'index_expr' or const_node.target != 'constant':
        return None
    
    split_value = const_node.args[1]
    
    # Find which index this references
    get_index_node = idx_expr_node.args[1]
    if get_index_node.op != 'call_module' or get_index_node.target != 'get_index':
        return None
    
    idx_key = get_index_node.args[0]
    idx_expr = body.indexing_exprs.get(idx_key)
    if idx_expr is None:
        return None
    
    iter_vars = body.iter_vars
    if len(iter_vars) == 0:
        return None
    innermost_var = iter_vars[-1]
    
    if idx_expr != innermost_var:
        return None
    
    return split_value, innermost_var


def make_half_inner_fn(original_fn, dim_idx, split_point, which_half):
    """Create inner_fn for one half of the split.
    
    The trick: offset i3 so that the branch condition becomes
    statically true/false. The ops.masked and ops.where will still
    be emitted, but Triton/codegen can optimize them.
    
    For which_half=0: i3 ranges [0, split_point), condition is True
    For which_half=1: i3 ranges [0, total-split_point), offset by split_point
    """
    def half_fn(idx):
        new_idx = list(idx)
        if which_half == 1:
            new_idx[dim_idx] = idx[dim_idx] + split_point
        return original_fn(new_idx)
    return half_fn


def split_pointwise_cat_node(scheduler, node):
    """Split a pointwise_cat SchedulerNode into two half-D nodes."""
    ir_node = node.node  # ComputedBuffer
    pw_data = ir_node.data  # Pointwise
    
    body = node._body
    result = detect_pointwise_cat_split(body)
    if result is None:
        return None
    
    split_point, innermost_var = result
    ranges = list(pw_data.ranges)
    dim_idx = len(ranges) - 1  # innermost
    
    # Verify split is clean
    total_dim = ranges[dim_idx]
    if not isinstance(total_dim, int):
        total_dim = int(total_dim)
    if not isinstance(split_point, int):
        split_point = int(split_point)
    
    second_half_size = total_dim - split_point
    
    print(f"  Splitting {node.get_name()}: [{','.join(str(r) for r in ranges)}] -> "
          f"2x [{','.join(str(r) if i != dim_idx else str(split_point) for i, r in enumerate(ranges))}]")
    
    # Create two new Pointwise IR nodes
    ranges_half = list(ranges)
    ranges_half[dim_idx] = split_point
    
    fn_lo = make_half_inner_fn(pw_data.inner_fn, dim_idx, split_point, 0)
    fn_hi = make_half_inner_fn(pw_data.inner_fn, dim_idx, split_point, 1)
    
    # Create output buffer (ConcatKernel-style)
    # We need to register these with the graph
    output_size = list(ranges)
    output_stride = ir.FlexibleLayout.contiguous_strides(output_size)
    
    # Create the output buffer
    concat_layout = ir.FixedLayout(
        device=pw_data.device,
        dtype=pw_data.dtype,
        size=output_size,
        stride=output_stride,
    )
    
    # Create two ComputedBuffers that write to slices
    # First: we create Pointwise nodes
    pw_lo = ir.Pointwise(
        device=pw_data.device,
        dtype=pw_data.dtype,
        inner_fn=fn_lo,
        ranges=ranges_half,
    )
    pw_lo._post_init_setattr("origin_node", pw_data.origin_node)
    
    ranges_half_hi = list(ranges)
    ranges_half_hi[dim_idx] = second_half_size
    
    pw_hi = ir.Pointwise(
        device=pw_data.device,
        dtype=pw_data.dtype,
        inner_fn=fn_hi,
        ranges=ranges_half_hi,
    )
    pw_hi._post_init_setattr("origin_node", pw_data.origin_node)
    
    # Create ComputedBuffers and register them
    # The store addresses need to be offset for the second half
    buf_lo = ir.ComputedBuffer(
        name=None,
        layout=ir.FixedLayout(
            device=pw_data.device,
            dtype=pw_data.dtype,
            size=ranges_half,
            stride=ir.FlexibleLayout.contiguous_strides(ranges_half),
        ),
        data=pw_lo,
    )
    buf_lo._post_init_setattr("origins", ir_node.origins)
    
    buf_hi = ir.ComputedBuffer(
        name=None,
        layout=ir.FixedLayout(
            device=pw_data.device,
            dtype=pw_data.dtype,
            size=ranges_half_hi,
            stride=ir.FlexibleLayout.contiguous_strides(ranges_half_hi),
        ),
        data=pw_hi,
    )
    buf_hi._post_init_setattr("origins", ir_node.origins)
    
    # Register with graph
    buf_lo_name = V.graph.register_buffer(buf_lo, set_name=True)
    buf_hi_name = V.graph.register_buffer(buf_hi, set_name=True)
    
    # Register as operations
    V.graph.register_operation(buf_lo)
    V.graph.register_operation(buf_hi)
    
    # Create scheduler nodes
    sn_lo = sched_mod.SchedulerNode(scheduler, buf_lo)
    sn_hi = sched_mod.SchedulerNode(scheduler, buf_hi)
    
    return sn_lo, sn_hi


def my_pre_fusion_pass(nodes):
    """Detect and split pointwise_cat nodes."""
    if not nodes:
        return nodes
    
    scheduler = nodes[0].scheduler
    new_nodes = []
    changed = False
    
    for node in nodes:
        if not isinstance(node, sched_mod.SchedulerNode):
            new_nodes.append(node)
            continue
        if not hasattr(node, '_body'):
            new_nodes.append(node)
            continue
        
        result = split_pointwise_cat_node(scheduler, node)
        if result is not None:
            sn_lo, sn_hi = result
            new_nodes.extend([sn_lo, sn_hi])
            changed = True
            print(f"  Split {node.get_name()} -> {sn_lo.get_name()}, {sn_hi.get_name()}")
        else:
            new_nodes.append(node)
    
    if changed:
        print(f"  Result: {len(new_nodes)} nodes (was {len(nodes)})")
    
    return new_nodes

cfg._pre_fusion_custom_pass = my_pre_fusion_pass

torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
try:
    with torch.no_grad():
        out = compiled(x, cos, sin)
    # Verify
    with torch.no_grad():
        ref = LlamaRoPE()(x, cos, sin)
    print(f"\nCorrectness: max_diff={((out - ref).abs().max().item()):.6f}")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
