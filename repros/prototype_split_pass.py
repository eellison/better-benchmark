"""Prototype: pre-fusion pass that splits pointwise_cat consumers on innermost dim.

The idea: when a SchedulerNode's LoopBody has a where(p3 < CONST, branch1, branch2)
pattern from pointwise_cat, and the computation in each branch is independent,
split it into two nodes with half the innermost dim, each writing to a slice of output.

This is NOT a pattern matcher for RoPE — it's a general optimization for any
pointwise_cat on innermost dim.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor import scheduler as sched_mod
import sympy
import time

cfg.force_disable_caches = True
cfg.prefer_concat_kernel_shared_reads = False  # Disable so we hit pointwise_cat path

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

def detect_innermost_cat_split(node):
    """Detect if a SchedulerNode has a pointwise_cat pattern on innermost dim.
    
    Returns (split_point, innermost_var_idx) if found, None otherwise.
    """
    if not isinstance(node, sched_mod.SchedulerNode):
        return None
    if not hasattr(node, '_body'):
        return None
    
    body = node._body
    
    # Check for subblocks (masked_subblock1, masked_subblock2 = pointwise_cat pattern)
    if 'masked_subblock1' not in body.subblocks or 'masked_subblock2' not in body.subblocks:
        return None
    
    # Must be exactly 2 subblocks (binary cat)
    masked_subblocks = [k for k in body.subblocks if k.startswith('masked_subblock')]
    if len(masked_subblocks) != 2:
        return None
    
    # Find the where() node and its lt condition
    root_graph = body.root_block.graph
    where_nodes = [n for n in root_graph.nodes if n.target == 'where']
    if len(where_nodes) != 1:
        return None
    
    where_node = where_nodes[0]
    # where(ops, condition, val_true, val_false)
    cond_node = where_node.args[1]  # the lt node
    
    # Check it's a lt comparison
    if cond_node.target != 'lt':
        return None
    
    # lt(ops, index_expr, constant)
    # The index_expr should reference the innermost variable
    lt_args = cond_node.args
    idx_expr_node = lt_args[1]  # index_expr node
    const_node = lt_args[2]  # constant node
    
    if idx_expr_node.target != 'index_expr' or const_node.target != 'constant':
        return None
    
    # Get the constant value (split point)
    split_value = const_node.args[1]  # constant(ops, value, dtype)
    
    # Get the index expression name
    idx_name_node = idx_expr_node.args[1]  # get_index result
    # Trace back to get_index call
    if idx_name_node.op != 'call_module' or idx_name_node.target != 'get_index':
        # Try going up the graph
        pass
    
    # Check which index it references
    idx_key = idx_name_node.args[0] if idx_name_node.op == 'call_module' else None
    if idx_key is None:
        return None
    
    # Check this is the innermost variable
    idx_expr = body.indexing_exprs.get(idx_key)
    if idx_expr is None:
        return None
    
    # The innermost var should be the last one (p3 for 4D)
    iter_vars = body.iter_vars
    if len(iter_vars) == 0:
        return None
    innermost_var = iter_vars[-1]
    
    # Check if idx_expr is just the innermost variable
    if idx_expr != innermost_var:
        return None
    
    # Get the innermost range
    sizes = body.sizes[0]  # iteration sizes
    innermost_size = sizes[-1]
    
    print(f"  DETECTED: split at {split_value} on innermost dim (size {innermost_size})")
    print(f"  Index expr '{idx_key}' = {idx_expr} (var {innermost_var})")
    
    return split_value, len(iter_vars) - 1

def pre_fusion_split_pass(nodes):
    """Pre-fusion pass: detect and report pointwise_cat split opportunities."""
    print(f"\n{'='*60}")
    print(f"Pre-fusion pass: scanning {len(nodes)} nodes")
    print(f"{'='*60}")
    
    for node in nodes:
        result = detect_innermost_cat_split(node)
        if result:
            split_val, dim_idx = result
            sizes = node._sizes[0]
            print(f"  Node {node.get_name()}: sizes={list(sizes)}")
            print(f"  Could split into 2 nodes of size {list(sizes[:-1]) + [split_val]}")
            print(f"  (saves {sizes[-1] - split_val} elements per iteration)")
    
    return nodes

cfg._pre_fusion_custom_pass = pre_fusion_split_pass

torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x, cos, sin)

# Verify correctness
with torch.no_grad():
    ref = LlamaRoPE()(x, cos, sin)
print(f"\nCorrectness: {torch.allclose(out, ref, atol=0.1)}")
