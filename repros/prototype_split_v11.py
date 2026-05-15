"""Prototype v11: Pure FX graph pass — no re-tracing.

Take the existing LoopBody, clone it, restrict var_ranges, run branch
elimination directly on the FX graph, use modified LoopBody as inner_fn.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.virtualized import V, ops
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody, LoopBodyBlock
from torch._inductor.sizevars import statically_known_true
import sympy
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._sympy.functions import Identity
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


def clone_loopbody(body: LoopBody, new_var_ranges=None, indexing_expr_transform=None):
    """Clone a LoopBody with optionally modified var_ranges and indexing_exprs.

    This avoids re-tracing — we just copy the existing FX graph structure.
    """
    import copy as _copy

    # LoopBody._init_with_copy expects a LoopBody as fn arg
    # But we can do a manual shallow copy
    new_body = LoopBody.__new__(LoopBody)

    vr = new_var_ranges if new_var_ranges else dict(body.var_ranges)
    new_body.var_ranges = vr
    new_body.iter_vars = list(body.iter_vars)
    new_body.reduce_vars = list(body.reduce_vars)
    new_body.sizes = (tuple(vr[v] for v in new_body.iter_vars), tuple())

    # Copy indexing_exprs, optionally transforming them
    # Always strip Identity wrappers — they prevent CSE from deduplicating loads
    def strip_identity(expr):
        return expr.replace(Identity, lambda x: x)

    if indexing_expr_transform:
        new_body.indexing_exprs = {
            k: strip_identity(indexing_expr_transform(k, v))
            for k, v in body.indexing_exprs.items()
        }
    else:
        new_body.indexing_exprs = {
            k: strip_identity(v) for k, v in body.indexing_exprs.items()
        }

    new_body.indirect_vars = list(body.indirect_vars)
    new_body.indirect_var_ranges = dict(body.indirect_var_ranges)
    new_body.memory_usage = body.memory_usage
    new_body.op_counts = body.op_counts
    new_body.has_partial_accumulate = body.has_partial_accumulate
    new_body.indexing = None

    # Clone submodules and subblocks
    new_body.submodules = {"get_index": new_body.get_index}
    new_body.subblocks = {}
    for name, block in body.subblocks.items():
        new_body.subblocks[name] = block.clone(new_body)
        new_body.submodules[name] = new_body.bind_masked_shim(name)

    # Clone the root block graph (shallow — shares Node objects until we modify)
    new_body.root_block = body.root_block.clone(new_body)
    # Deep copy the graph so modifications don't affect original
    new_body.root_block.graph = _copy.deepcopy(body.root_block.graph)

    return new_body


def eliminate_dead_branches(body: LoopBody) -> bool:
    """Eliminate statically-dead branches in a LoopBody's root graph."""
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
    where_nodes = [n for n in graph.nodes if n.op == "call_method" and n.target == "where"]

    for where_node in where_nodes:
        cond_node = where_node.args[1]
        true_node = where_node.args[2]
        false_node = where_node.args[3]

        cond_expr = _extract_comparison_expr(cond_node, body)
        if cond_expr is None:
            continue

        is_true = statically_known_true(shape_env, cond_expr, axioms, var_to_range)
        is_false = statically_known_true(shape_env, ~cond_expr, axioms, var_to_range)

        if not is_true and not is_false:
            continue

        live_node = true_node if is_true else false_node
        print(f"  Branch elimination: {cond_expr} → {'True' if is_true else 'False'}")

        # Inline the masked subblock if the live branch is a call_module
        if live_node.op == "call_module" and live_node.target.startswith("masked_subblock"):
            subblock_name = live_node.target
            subblock = body.subblocks.get(subblock_name)
            if subblock is not None:
                inlined = _inline_subblock(graph, where_node, subblock, body)
                if inlined is not None:
                    where_node.replace_all_uses_with(inlined)
                    graph.erase_node(where_node)
                    changed = True
                    continue

        # Fallback: replace where with live branch directly
        where_node.replace_all_uses_with(live_node)
        graph.erase_node(where_node)
        changed = True

    if changed:
        # Manual DCE
        dce_pass = True
        while dce_pass:
            dce_pass = False
            for node in reversed(list(graph.nodes)):
                if node.op in ("placeholder", "output"):
                    continue
                if len(node.users) == 0:
                    graph.erase_node(node)
                    dce_pass = True

    return changed


def _extract_comparison_expr(cond_node, body: LoopBody):
    """Extract sympy comparison from a condition node."""
    if cond_node.op != "call_method" or cond_node.target not in ("lt", "le", "gt", "ge"):
        return None

    lhs_expr = _node_to_sympy(cond_node.args[1], body)
    rhs_expr = _node_to_sympy(cond_node.args[2], body)
    if lhs_expr is None or rhs_expr is None:
        return None

    op_map = {"lt": sympy.Lt, "le": sympy.Le, "gt": sympy.Gt, "ge": sympy.Ge}
    return op_map[cond_node.target](lhs_expr, rhs_expr)


def _node_to_sympy(node, body: LoopBody):
    """Convert FX node to sympy expression."""
    if node.op != "call_method":
        return None
    if node.target == "index_expr":
        index_node = node.args[1]
        if index_node.op == "call_module" and index_node.target == "get_index":
            index_name = index_node.args[0]
            if index_name in body.indexing_exprs:
                return body.indexing_exprs[index_name]
    elif node.target == "constant":
        return sympy.Integer(node.args[1])
    return None


def _inline_subblock(graph, insert_before, subblock: LoopBodyBlock, body: LoopBody):
    """Inline subblock's FX graph nodes into root graph."""
    subgraph = subblock.graph

    ops_node = None
    output_node = None
    for node in subgraph.nodes:
        if node.op == "placeholder":
            ops_node = node
        if node.op == "output":
            output_node = node
    if not ops_node or not output_node:
        return None

    return_node = output_node.args[0]

    root_ops = None
    for node in graph.nodes:
        if node.op == "placeholder":
            root_ops = node
            break

    node_map = {ops_node: root_ops}
    with graph.inserting_before(insert_before):
        for node in subgraph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            new_args = tuple(
                node_map.get(a, a) if isinstance(a, torch.fx.Node) else a
                for a in node.args
            )
            new_kwargs = {
                k: node_map.get(v, v) if isinstance(v, torch.fx.Node) else v
                for k, v in node.kwargs.items()
            }
            new_node = graph.create_node(node.op, node.target, new_args, new_kwargs)
            node_map[node] = new_node

    return node_map.get(return_node)


# ============================================================
# Main pass
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

        # Detect split point from lt comparison
        dim_idx = ndim - 1
        split_point = None
        for node in body.root_block.graph.nodes:
            if node.op == "call_method" and node.target == "lt":
                rhs = node.args[2]
                if rhs.op == "call_method" and rhs.target == "constant":
                    split_point = int(rhs.args[1])
                    break
        if split_point is None:
            continue

        total_dim = int(ranges[dim_idx])
        d_var = iter_vars[dim_idx]
        orig_buf_name = op.get_name()
        print(f"\nSplit {orig_buf_name}: dim={dim_idx}, split={split_point}, total={total_dim}")

        # --- Half LO: d3 ∈ [0, split_point) ---
        var_ranges_lo = dict(var_ranges)
        var_ranges_lo[d_var] = split_point
        body_lo = clone_loopbody(body, new_var_ranges=var_ranges_lo)
        eliminate_dead_branches(body_lo)
        print(f"  LO graph:\n  {body_lo.root_block.debug_str('lo')}")

        # --- Half HI: d3 ∈ [0, total-split), with offset in indexing_exprs ---
        var_ranges_hi = dict(var_ranges)
        var_ranges_hi[d_var] = total_dim - split_point

        def offset_transform(name, expr, _d=d_var, _off=split_point):
            # Add offset to all indexing_exprs that use d_var
            if _d in expr.free_symbols:
                return expr.subs(_d, _d + _off)
            return expr

        body_hi = clone_loopbody(body, new_var_ranges=var_ranges_hi,
                                  indexing_expr_transform=offset_transform)
        eliminate_dead_branches(body_hi)
        print(f"  HI graph:\n  {body_hi.root_block.debug_str('hi')}")

        # Build inner_fns from the modified LoopBodies
        # inner_fn receives a list/tuple of sympy indices like [q0, q1, q2, q3]
        # LoopBody.__call__ expects (iter_indices,) where iter_indices is a flat list
        def make_inner_fn(body_obj):
            def inner_fn(idx):
                return body_obj(idx)
            return inner_fn

        fn_lo = make_inner_fn(body_lo)
        fn_hi = make_inner_fn(body_hi)

        # Build ConcatKernel
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
        print(f"  Created ConcatKernel: {buf_lo.get_name()}, {buf_hi.get_name()}")
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

# Benchmark
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
print(f"Split+eliminate: {t:.1f} us")

# Baseline
torch._dynamo.reset()
GraphLowering._update_scheduler = original_update
compiled_base = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    compiled_base(x_t, cos_t, sin_t)
t_base = bench_graph(compiled_base, [x_t, cos_t, sin_t])
print(f"Baseline: {t_base:.1f} us")
print(f"Speedup: {t_base/t:.2f}x")
