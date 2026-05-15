"""Test: Does split+branch-elimination compose with ConcatKernel fusion?

We want the two halves (buf1, buf2) to fuse into 1 kernel via
_score_shared_output_storage. Also test with Q+K RoPE (GQA case).
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
from torch.utils._sympy.functions import Identity
import sympy
from torch.utils._sympy.value_ranges import ValueRanges
import time
import copy
cfg.force_disable_caches = True

B, S, H, D = 4, 512, 32, 128
D_HALF = D // 2


class GQA_QK_RoPE(torch.nn.Module):
    """Q and K RoPE — the real GQA case with different head counts."""
    def __init__(self):
        super().__init__()
        self.h_q = 32
        self.h_kv = 8

    def forward(self, q, k, cos, sin):
        # q: [B, H_q, S, D], k: [B, H_kv, S, D]
        cos_e = cos.unsqueeze(0).unsqueeze(0)
        sin_e = sin.unsqueeze(0).unsqueeze(0)

        # Q RoPE
        q2 = q[..., D_HALF:]
        q1 = q[..., :D_HALF]
        q_rot = torch.cat([-q2, q1], dim=-1)
        q_out = q * cos_e + q_rot * sin_e

        # K RoPE
        k2 = k[..., D_HALF:]
        k1 = k[..., :D_HALF]
        k_rot = torch.cat([-k2, k1], dim=-1)
        k_out = k * cos_e + k_rot * sin_e

        return q_out, k_out


# --- Reuse the graph pass from v11 ---

def clone_loopbody(body, new_var_ranges=None, indexing_expr_transform=None):
    import copy as _copy
    new_body = LoopBody.__new__(LoopBody)
    vr = new_var_ranges if new_var_ranges else dict(body.var_ranges)
    new_body.var_ranges = vr
    new_body.iter_vars = list(body.iter_vars)
    new_body.reduce_vars = list(body.reduce_vars)
    new_body.sizes = (tuple(vr[v] for v in new_body.iter_vars), tuple())

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

    new_body.submodules = {"get_index": new_body.get_index}
    new_body.subblocks = {}
    for name, block in body.subblocks.items():
        new_body.subblocks[name] = block.clone(new_body)
        new_body.submodules[name] = new_body.bind_masked_shim(name)

    new_body.root_block = body.root_block.clone(new_body)
    new_body.root_block.graph = _copy.deepcopy(body.root_block.graph)
    return new_body

import copy as _copy

def eliminate_dead_branches(body):
    graph = body.root_block.graph
    shape_env = V.graph.sizevars.shape_env
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
        if live_node.op == "call_module" and live_node.target.startswith("masked_subblock"):
            subblock = body.subblocks.get(live_node.target)
            if subblock:
                inlined = _inline_subblock(graph, where_node, subblock, body)
                if inlined:
                    where_node.replace_all_uses_with(inlined)
                    graph.erase_node(where_node)
                    changed = True
                    continue
        where_node.replace_all_uses_with(live_node)
        graph.erase_node(where_node)
        changed = True

    if changed:
        dce = True
        while dce:
            dce = False
            for node in reversed(list(graph.nodes)):
                if node.op in ("placeholder", "output"):
                    continue
                if len(node.users) == 0:
                    graph.erase_node(node)
                    dce = True
    return changed


def _extract_comparison_expr(cond_node, body):
    if cond_node.op != "call_method" or cond_node.target not in ("lt", "le", "gt", "ge"):
        return None
    lhs = _node_to_sympy(cond_node.args[1], body)
    rhs = _node_to_sympy(cond_node.args[2], body)
    if lhs is None or rhs is None:
        return None
    op_map = {"lt": sympy.Lt, "le": sympy.Le, "gt": sympy.Gt, "ge": sympy.Ge}
    return op_map[cond_node.target](lhs, rhs)


def _node_to_sympy(node, body):
    if node.op != "call_method":
        return None
    if node.target == "index_expr":
        idx_node = node.args[1]
        if idx_node.op == "call_module" and idx_node.target == "get_index":
            name = idx_node.args[0]
            if name in body.indexing_exprs:
                return body.indexing_exprs[name]
    elif node.target == "constant":
        return sympy.Integer(node.args[1])
    return None


def _inline_subblock(graph, insert_before, subblock, body):
    subgraph = subblock.graph
    ops_node = output_node = None
    for node in subgraph.nodes:
        if node.op == "placeholder": ops_node = node
        if node.op == "output": output_node = node
    if not ops_node or not output_node:
        return None
    return_node = output_node.args[0]
    root_ops = next(n for n in graph.nodes if n.op == "placeholder")
    node_map = {ops_node: root_ops}
    with graph.inserting_before(insert_before):
        for node in subgraph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            new_args = tuple(node_map.get(a, a) if isinstance(a, torch.fx.Node) else a for a in node.args)
            new_kwargs = {k: node_map.get(v, v) if isinstance(v, torch.fx.Node) else v for k, v in node.kwargs.items()}
            node_map[node] = graph.create_node(node.op, node.target, new_args, new_kwargs)
    return node_map.get(return_node)


def split_pointwise_cat(self_graph, op):
    """Split a pointwise_cat ComputedBuffer into ConcatKernel with branch elimination."""
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
        return False
    masked = [k for k in body.subblocks if k.startswith('masked_subblock')]
    if len(masked) != 2:
        return False

    # Detect split point
    dim_idx = ndim - 1
    split_point = None
    for node in body.root_block.graph.nodes:
        if node.op == "call_method" and node.target == "lt":
            rhs = node.args[2]
            if rhs.op == "call_method" and rhs.target == "constant":
                split_point = int(rhs.args[1])
                break
    if split_point is None:
        return False

    total_dim = int(ranges[dim_idx])
    d_var = iter_vars[dim_idx]
    orig_buf_name = op.get_name()

    # Half LO
    var_ranges_lo = dict(var_ranges)
    var_ranges_lo[d_var] = split_point
    body_lo = clone_loopbody(body, new_var_ranges=var_ranges_lo)
    eliminate_dead_branches(body_lo)

    # Half HI
    var_ranges_hi = dict(var_ranges)
    var_ranges_hi[d_var] = total_dim - split_point
    def offset_transform(name, expr, _d=d_var, _off=split_point):
        if _d in expr.free_symbols:
            return expr.subs(_d, _d + _off)
        return expr
    body_hi = clone_loopbody(body, new_var_ranges=var_ranges_hi, indexing_expr_transform=offset_transform)
    eliminate_dead_branches(body_hi)

    # Build inner_fns
    def make_inner_fn(body_obj):
        def inner_fn(idx):
            return body_obj(idx)
        return inner_fn

    fn_lo = make_inner_fn(body_lo)
    fn_hi = make_inner_fn(body_hi)

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
    self_graph.name_to_buffer[orig_buf_name] = concat_kernel
    self_graph.register_operation(concat_kernel)
    self_graph.operations.remove(op)
    print(f"  Split {orig_buf_name} -> {buf_lo.get_name()}, {buf_hi.get_name()}")
    return True


# --- Patch scheduler with BOTH fixes ---

original_update = GraphLowering._update_scheduler

def patched_update_scheduler(self):
    from torch._inductor.scheduler import Scheduler, SchedulerNode

    # Pass 1: split all pointwise_cat ops
    split_any = True
    while split_any:
        split_any = False
        for op in list(self.operations):
            if isinstance(op, ir.ComputedBuffer) and isinstance(op.data, ir.Pointwise):
                if split_pointwise_cat(self, op):
                    split_any = True
                    break

    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)

    # Patch score_fusion_memory to handle shared output storage
    original_score = self.scheduler.score_fusion_memory.__func__

    def patched_score(sched_self, node1, node2, **kwargs):
        score = original_score(sched_self, node1, node2, **kwargs)
        if score == 0:
            score = _score_shared_output_storage(sched_self, node1, node2)
        return score

    import types
    self.scheduler.score_fusion_memory = types.MethodType(patched_score, self.scheduler)

    # Print kernel count
    print(f"  Scheduler: {len(self.scheduler.nodes)} nodes -> ", end="")
    fused = self.scheduler.fuse_nodes(self.scheduler.nodes)
    print(f"{len(fused)} fused groups")


def _score_shared_output_storage(sched, node1, node2):
    """Score fusion for nodes that write to the same underlying storage (ConcatKernel siblings)."""
    from torch._inductor.scheduler import SchedulerNode

    def _get_storage_name(node):
        if not isinstance(node, SchedulerNode):
            return None
        ir_node = node.node
        if not hasattr(ir_node, "layout"):
            return None
        layout = ir_node.layout
        if not isinstance(layout, ir.NonOwningLayout):
            return None
        view = layout.view
        if isinstance(view, ir.ReinterpretView):
            storage = view.data
            if isinstance(storage, ir.StorageBox) and hasattr(storage.data, "get_name"):
                return storage.data.get_name()
        return None

    name1 = _get_storage_name(node1)
    name2 = _get_storage_name(node2)
    if name1 is not None and name1 == name2:
        return sum(sched.dep_size_hint(dep) for dep in node1.read_writes.writes)
    return 0


GraphLowering._update_scheduler = patched_update_scheduler

# --- Test single RoPE ---
print("=" * 60)
print("Test 1: Single RoPE (should be 1 kernel, 6.2 us)")
print("=" * 60)

class SingleRoPE(torch.nn.Module):
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

torch._dynamo.reset()
compiled = torch.compile(SingleRoPE().cuda())
with torch.no_grad():
    out = compiled(x_t, cos_t, sin_t)
    ref = SingleRoPE()(x_t, cos_t, sin_t)
print(f"  Correctness: max_diff={((out - ref).abs().max().item()):.6f}")


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
print(f"  Time: {t:.1f} us")

# --- Test Q+K RoPE (GQA) ---
print("\n" + "=" * 60)
print("Test 2: GQA Q+K RoPE (should fuse Q and K halves)")
print("=" * 60)

q_t = torch.randn(B, 32, S, D, dtype=torch.bfloat16, device='cuda')
k_t = torch.randn(B, 8, S, D, dtype=torch.bfloat16, device='cuda')

torch._dynamo.reset()
compiled_gqa = torch.compile(GQA_QK_RoPE().cuda())
with torch.no_grad():
    q_out, k_out = compiled_gqa(q_t, k_t, cos_t, sin_t)
    q_ref, k_ref = GQA_QK_RoPE()(q_t, k_t, cos_t, sin_t)
print(f"  Correctness Q: max_diff={((q_out - q_ref).abs().max().item()):.6f}")
print(f"  Correctness K: max_diff={((k_out - k_ref).abs().max().item()):.6f}")

t_gqa = bench_graph(compiled_gqa, [q_t, k_t, cos_t, sin_t])
print(f"  Time: {t_gqa:.1f} us")

# Baseline
GraphLowering._update_scheduler = original_update
torch._dynamo.reset()
compiled_base = torch.compile(GQA_QK_RoPE().cuda())
with torch.no_grad():
    compiled_base(q_t, k_t, cos_t, sin_t)
t_base = bench_graph(compiled_base, [q_t, k_t, cos_t, sin_t])
print(f"  Baseline: {t_base:.1f} us")
print(f"  Speedup: {t_base/t_gqa:.2f}x")
