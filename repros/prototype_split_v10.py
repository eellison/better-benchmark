"""Prototype v10: Generic branch elimination via LoopBody graph pass.

Instead of hardcoding clean inner_fns, we:
1. Split pointwise_cat into two halves (as before)
2. For each half, trace the ORIGINAL inner_fn with restricted var_ranges into a LoopBody
3. Run a graph pass on the LoopBody that eliminates dead branches:
   - Find `where` nodes whose condition is statically provable
   - Replace `where` with the live branch result
   - Inline the masked subblock's graph into the root graph
   - DCE dead nodes
4. Reconstruct a clean inner_fn from the simplified LoopBody

This is the generic version of what prototype_split_v9 does with hardcoded inner_fns.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.virtualized import V, ops
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody, LoopBodyBlock, InterpreterShim
from torch._inductor.sizevars import statically_known_true
import sympy
from sympy import S
from torch.utils._sympy.value_ranges import ValueRanges
import time
import copy
cfg.force_disable_caches = True

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


def eliminate_dead_branches(body: LoopBody) -> bool:
    """Eliminate statically-dead branches in a LoopBody's root graph.

    When a `where(cond, true_val, false_val)` node has a condition that's
    statically provable given var_ranges, replace it with the live branch
    and inline the corresponding masked_subblock.

    Returns True if any branches were eliminated.
    """
    graph = body.root_block.graph
    shape_env = V.graph.sizevars.shape_env

    # Build axioms and var_to_range from body's var_ranges
    axioms = []
    var_to_range = []
    for var, size in body.var_ranges.items():
        axioms.append(0 <= var)
        axioms.append(var < size)
        var_to_range.append((var, ValueRanges(0, int(size) - 1)))
    axioms = tuple(axioms) + shape_env.get_axioms()
    var_to_range = tuple(var_to_range)

    changed = False

    # Find where nodes and check if condition is statically known
    where_nodes = [n for n in graph.nodes if n.op == "call_method" and n.target == "where"]

    for where_node in where_nodes:
        # where(ops, cond, true_val, false_val)
        # In the FX graph: args = (ops_proxy, cond, true_val, false_val)
        cond_node = where_node.args[1]
        true_node = where_node.args[2]
        false_node = where_node.args[3]

        # Try to resolve the condition statically.
        # The condition comes from ops.lt(ops.index_expr(d3), ops.constant(64))
        # We need to trace back to find the sympy expression.
        cond_expr = _extract_comparison_expr(cond_node, body)
        if cond_expr is None:
            continue

        # Check if condition is statically True or False
        is_true = statically_known_true(shape_env, cond_expr, axioms, var_to_range)
        is_false = statically_known_true(shape_env, ~cond_expr, axioms, var_to_range)

        if not is_true and not is_false:
            continue

        live_node = true_node if is_true else false_node
        dead_node = false_node if is_true else true_node

        print(f"  Branch elimination: {cond_expr} is statically {'True' if is_true else 'False'}")

        # The live_node is a call_module to a masked_subblock (e.g., masked_subblock1)
        # We need to inline its computation into the root graph.
        if live_node.op == "call_module":
            subblock_name = live_node.target
            subblock = body.subblocks.get(subblock_name)
            if subblock is not None:
                # Inline the subblock's graph into the root graph
                inlined_result = _inline_subblock(graph, where_node, subblock, body)
                if inlined_result is not None:
                    where_node.replace_all_uses_with(inlined_result)
                    graph.erase_node(where_node)
                    changed = True
                    print(f"  Inlined {subblock_name} into root graph")
                    continue

        # Fallback: just replace where with the live branch value directly
        where_node.replace_all_uses_with(live_node)
        graph.erase_node(where_node)
        changed = True

    if changed:
        # Manual DCE: remove nodes with no users (except output)
        # Iterate in reverse to handle chains
        changed_dce = True
        while changed_dce:
            changed_dce = False
            for node in reversed(list(graph.nodes)):
                if node.op in ("placeholder", "output"):
                    continue
                if len(node.users) == 0:
                    graph.erase_node(node)
                    changed_dce = True

    return changed


def _extract_comparison_expr(cond_node, body: LoopBody):
    """Extract a sympy comparison expression from a condition node.

    Pattern: ops.lt(ops.index_expr(index_name, dtype), ops.constant(val, dtype))
    We resolve the index_expr to its sympy expression from body.indexing_exprs.
    """
    if cond_node.op != "call_method":
        return None

    if cond_node.target not in ("lt", "le", "gt", "ge"):
        return None

    # args = (ops, lhs, rhs)
    lhs_node = cond_node.args[1]
    rhs_node = cond_node.args[2]

    lhs_expr = _node_to_sympy(lhs_node, body)
    rhs_expr = _node_to_sympy(rhs_node, body)

    if lhs_expr is None or rhs_expr is None:
        return None

    if cond_node.target == "lt":
        return sympy.Lt(lhs_expr, rhs_expr)
    elif cond_node.target == "le":
        return sympy.Le(lhs_expr, rhs_expr)
    elif cond_node.target == "gt":
        return sympy.Gt(lhs_expr, rhs_expr)
    elif cond_node.target == "ge":
        return sympy.Ge(lhs_expr, rhs_expr)
    return None


def _node_to_sympy(node, body: LoopBody):
    """Convert an FX node to a sympy expression, if possible."""
    if node.op == "call_method":
        if node.target == "index_expr":
            # args = (ops, index_val, dtype)
            # index_val is from get_index which returns a sympy expr
            index_node = node.args[1]
            if index_node.op == "call_module" and index_node.target == "get_index":
                # get_index(name) -> body.indexing_exprs[name]
                index_name = index_node.args[0]
                if index_name in body.indexing_exprs:
                    return body.indexing_exprs[index_name]
            return None
        elif node.target == "constant":
            # args = (ops, value, dtype)
            return sympy.Integer(node.args[1])
    return None


def _inline_subblock(graph, insert_before, subblock: LoopBodyBlock, body: LoopBody):
    """Inline a subblock's FX graph nodes into the root graph.

    The subblock graph looks like:
        ops = placeholder
        get_index = self.get_index('index3')
        load = ops.load('arg2_1', get_index)
        neg = ops.neg(load)
        output(neg)

    We need to transplant the computation nodes (get_index, load, neg) into the
    root graph, wiring up the ops placeholder to the root's ops.
    """
    subgraph = subblock.graph

    # Find the ops placeholder and output in subblock
    ops_node = None
    output_node = None
    for node in subgraph.nodes:
        if node.op == "placeholder":
            ops_node = node
        if node.op == "output":
            output_node = node

    if ops_node is None or output_node is None:
        return None

    # The output's arg is the return value
    return_node = output_node.args[0]

    # Find the root graph's ops placeholder
    root_ops = None
    for node in graph.nodes:
        if node.op == "placeholder":
            root_ops = node
            break

    if root_ops is None:
        return None

    # Clone subblock nodes into root graph, remapping references
    node_map = {ops_node: root_ops}

    with graph.inserting_before(insert_before):
        for node in subgraph.nodes:
            if node.op == "placeholder" or node.op == "output":
                continue

            # Remap args
            new_args = tuple(
                node_map.get(a, a) if isinstance(a, torch.fx.Node) else a
                for a in node.args
            )
            new_kwargs = {
                k: node_map.get(v, v) if isinstance(v, torch.fx.Node) else v
                for k, v in node.kwargs.items()
            }

            # Use create_node directly to avoid module validation
            new_node = graph.create_node(node.op, node.target, new_args, new_kwargs)
            node_map[node] = new_node

    return node_map.get(return_node)


# ============================================================
# Test the pass
# ============================================================

x_t = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos_t = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')
sin_t = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')

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
            sym = sympy.Symbol(f"d{j}", integer=True, nonneg=True)
            var_ranges[sym] = r
            iter_vars.append(sym)
        try:
            body = LoopBody(pw.inner_fn, (iter_vars,), var_ranges, iter_vars, [])
        except Exception:
            continue
        masked_subblocks = [k for k in body.subblocks if k.startswith('masked_subblock')]
        if len(masked_subblocks) != 2:
            continue

        # Found pointwise_cat pattern. Now split and eliminate branches.
        dim_idx = ndim - 1
        split_point = None

        # Detect split point from the `lt` comparison in the graph
        for node in body.root_block.graph.nodes:
            if node.op == "call_method" and node.target == "lt":
                rhs = node.args[2]
                if rhs.op == "call_method" and rhs.target == "constant":
                    split_point = int(rhs.args[1])
                    break

        if split_point is None:
            continue

        total_dim = int(ranges[dim_idx])
        orig_buf_name = op.get_name()

        print(f"\nSplitting {orig_buf_name} at dim {dim_idx}, split={split_point}, total={total_dim}")

        # Create two halves with restricted var_ranges
        # Half lo: d3 ∈ [0, split_point)
        var_ranges_lo = dict(var_ranges)
        var_ranges_lo[iter_vars[dim_idx]] = split_point

        # Half hi: d3 ∈ [0, total_dim - split_point)
        # But inner_fn uses d3 starting from split_point, so we need offset
        var_ranges_hi = dict(var_ranges)
        var_ranges_hi[iter_vars[dim_idx]] = total_dim - split_point

        # For half_lo: trace with restricted ranges and eliminate dead branches
        print("\n  --- Half LO (d3 ∈ [0, 64)) ---")
        body_lo = LoopBody(pw.inner_fn, (iter_vars,), var_ranges_lo, iter_vars, [])
        print(f"  Before elimination:")
        print(f"  {body_lo.root_block.debug_str('lo')}")
        eliminated = eliminate_dead_branches(body_lo)
        if eliminated:
            print(f"  After elimination:")
            print(f"  {body_lo.root_block.debug_str('lo')}")

        # For half_hi: need to offset d3 by split_point in the inner_fn
        # Create an offset inner_fn
        def make_hi_fn(original_fn, dim_idx, split_point):
            def hi_fn(idx):
                idx_list = list(idx)
                idx_list[dim_idx] = idx_list[dim_idx] + split_point
                return original_fn(idx_list)
            return hi_fn

        hi_fn = make_hi_fn(pw.inner_fn, dim_idx, split_point)
        print(f"\n  --- Half HI (d3 ∈ [0, 64), offset by {split_point}) ---")
        body_hi = LoopBody(hi_fn, (iter_vars,), var_ranges_hi, iter_vars, [])
        print(f"  Before elimination:")
        print(f"  {body_hi.root_block.debug_str('hi')}")
        eliminated = eliminate_dead_branches(body_hi)
        if eliminated:
            print(f"  After elimination:")
            print(f"  {body_hi.root_block.debug_str('hi')}")

        # Now reconstruct inner_fns from the simplified LoopBodies
        # LoopBody.__call__(indices) executes the graph with the ops handler
        def make_inner_fn_from_body(body_obj):
            def inner_fn(idx):
                return body_obj((idx,))
            return inner_fn

        fn_lo = make_inner_fn_from_body(body_lo)
        fn_hi = make_inner_fn_from_body(body_hi)

        # Build the ConcatKernel
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
        print(f"\n  Split {orig_buf_name} -> {buf_lo.get_name()}, {buf_hi.get_name()}")
        break

    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)


GraphLowering._update_scheduler = patched_update_scheduler
torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x_t, cos_t, sin_t)
    ref = LlamaRoPE()(x_t, cos_t, sin_t)

print(f"\nCorrectness: max_diff={((out - ref).abs().max().item()):.6f}")
GraphLowering._update_scheduler = original_update
