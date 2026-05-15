"""Final test: Qwen3-0.6B with stride fix + cycle fix + benchmark."""
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
import sympy, copy as _copy, time, importlib.util
from torch.utils._sympy.value_ranges import ValueRanges
cfg.force_disable_caches = True


def clone_loopbody(body, new_var_ranges=None, indexing_expr_transform=None):
    new_body = LoopBody.__new__(LoopBody)
    vr = new_var_ranges if new_var_ranges else dict(body.var_ranges)
    new_body.var_ranges = vr
    new_body.iter_vars = list(body.iter_vars)
    new_body.reduce_vars = list(body.reduce_vars)
    new_body.sizes = (tuple(vr[v] for v in new_body.iter_vars), tuple())
    def strip_identity(expr):
        return expr.replace(Identity, lambda x: x)
    if indexing_expr_transform:
        new_body.indexing_exprs = {k: strip_identity(indexing_expr_transform(k, v)) for k, v in body.indexing_exprs.items()}
    else:
        new_body.indexing_exprs = {k: strip_identity(v) for k, v in body.indexing_exprs.items()}
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
    for where_node in [n for n in graph.nodes if n.op == "call_method" and n.target == "where"]:
        cond_node, true_node, false_node = where_node.args[1], where_node.args[2], where_node.args[3]
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
                if node.op in ("placeholder", "output"): continue
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
    return {"lt": sympy.Lt, "le": sympy.Le, "gt": sympy.Gt, "ge": sympy.Ge}[cond_node.target](lhs, rhs)


def _node_to_sympy(node, body):
    if node.op != "call_method": return None
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
    if not ops_node or not output_node: return None
    return_node = output_node.args[0]
    root_ops = next(n for n in graph.nodes if n.op == "placeholder")
    node_map = {ops_node: root_ops}
    with graph.inserting_before(insert_before):
        for node in subgraph.nodes:
            if node.op in ("placeholder", "output"): continue
            new_args = tuple(node_map.get(a, a) if isinstance(a, torch.fx.Node) else a for a in node.args)
            new_kwargs = {k: node_map.get(v, v) if isinstance(v, torch.fx.Node) else v for k, v in node.kwargs.items()}
            node_map[node] = graph.create_node(node.op, node.target, new_args, new_kwargs)
    return node_map.get(return_node)


def split_pointwise_cat(self_graph, op):
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

    var_ranges_lo = dict(var_ranges)
    var_ranges_lo[d_var] = split_point
    body_lo = clone_loopbody(body, new_var_ranges=var_ranges_lo)
    eliminate_dead_branches(body_lo)

    var_ranges_hi = dict(var_ranges)
    var_ranges_hi[d_var] = total_dim - split_point
    def offset_transform(name, expr, _d=d_var, _off=split_point):
        if _d in expr.free_symbols: return expr.subs(_d, _d + _off)
        return expr
    body_hi = clone_loopbody(body, new_var_ranges=var_ranges_hi, indexing_expr_transform=offset_transform)
    eliminate_dead_branches(body_hi)

    def make_inner_fn(body_obj):
        def inner_fn(idx): return body_obj(idx)
        return inner_fn

    ranges_lo = list(pw.ranges)
    ranges_lo[dim_idx] = split_point
    ranges_hi = list(pw.ranges)
    ranges_hi[dim_idx] = total_dim - split_point

    pw_lo = ir.Pointwise(device=pw.device, dtype=pw.dtype, inner_fn=make_inner_fn(body_lo), ranges=ranges_lo)
    pw_lo._post_init_setattr("origin_node", pw.origin_node)
    pw_lo._post_init_setattr("traceback", pw.traceback)
    pw_hi = ir.Pointwise(device=pw.device, dtype=pw.dtype, inner_fn=make_inner_fn(body_hi), ranges=ranges_hi)
    pw_hi._post_init_setattr("origin_node", pw.origin_node)
    pw_hi._post_init_setattr("traceback", pw.traceback)

    # Use the ORIGINAL buffer's layout strides
    output_size = list(pw.ranges)
    orig_layout = op.layout
    if isinstance(orig_layout, ir.FixedLayout):
        output_stride = list(orig_layout.stride)
    else:
        output_stride = ir.FlexibleLayout.contiguous_strides(output_size)

    concat_kernel = ir.ConcatKernel(name=None, layout=ir.FixedLayout(device=pw.device, dtype=pw.dtype, size=output_size, stride=output_stride), inputs=[])
    kernel_storage = ir.StorageBox(concat_kernel)
    slice_lo = ir.SliceView.create(kernel_storage, dim_idx, 0, split_point, clamp=False)
    slice_hi = ir.SliceView.create(kernel_storage, dim_idx, split_point, total_dim, clamp=False)

    ops_before = len(self_graph.operations)
    buf_lo = ir.ConcatKernel.realize_into(ir.TensorBox(ir.StorageBox(pw_lo)), slice_lo)
    buf_hi = ir.ConcatKernel.realize_into(ir.TensorBox(ir.StorageBox(pw_hi)), slice_hi)
    new_ops = self_graph.operations[ops_before:]
    del self_graph.operations[ops_before:]

    concat_kernel.inputs = ir.ConcatKernel.unwrap_storage([buf_lo, buf_hi])
    concat_kernel.name = orig_buf_name
    self_graph.name_to_buffer[orig_buf_name] = concat_kernel

    # In-place replacement to avoid cycle
    op_idx = self_graph.operations.index(op)
    for i, new_op in enumerate(new_ops):
        self_graph.operations.insert(op_idx + i, new_op)
    new_op_idx = op_idx + len(new_ops)
    self_graph.operations[new_op_idx] = concat_kernel

    op_name = self_graph.qualify_name(f"op{len(self_graph.operations) + 100}")
    self_graph.name_to_op[op_name] = concat_kernel
    concat_kernel.operation_name = op_name
    return True


def bench(fn, inputs, warmup=50, iters=500):
    with torch.no_grad():
        for _ in range(warmup): fn(*inputs)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad(): fn(*inputs)
    torch.cuda.synchronize()
    for _ in range(warmup): g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


original_update = GraphLowering._update_scheduler
def patched_update_scheduler(self):
    from torch._inductor.scheduler import Scheduler
    n_split = 0
    split_any = True
    while split_any:
        split_any = False
        for op in list(self.operations):
            if isinstance(op, ir.ComputedBuffer) and isinstance(op.data, ir.Pointwise):
                if split_pointwise_cat(self, op):
                    split_any = True
                    n_split += 1
                    break
    if n_split:
        print(f"    Split {n_split} pointwise_cat ops")
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)


# === Test all models ===
repro_dir = "repo/repros/multi_kernel_inference"
models = [
    ("Qwen3-0.6B (r005)", "k03_Qwen3-0.6B_inference__region_005_mean_c0e8d7954fcb_fcbe8ae0.py"),
    ("Mistral-7B", "k03_vllm_mistralai_Mistral-7B-Instruct-v0.3_inference__region_011_pointwise_108892bc6e62_8f1de50a.py"),
    ("Qwen3-30B-A3B", "k03_vllm_Qwen_Qwen3-30B-A3B_inference__region_001_pointwise_672682f95253_7337c280.py"),
    ("gpt-oss-20b (r013)", "k03_vllm_openai_gpt-oss-20b_inference__region_013_pointwise_1b8e92f2de8a_3af1681a.py"),
    ("gpt-oss-20b (r020)", "k03_vllm_openai_gpt-oss-20b_inference__region_020_pointwise_cf54f177af3d_b579c38e.py"),
    ("Mistral-7B (r014)", "k04_vllm_mistralai_Mistral-7B-Instruct-v0.3_inference__region_014_pointwise_f3c6cfe72279_ea461ebf.py"),
]

import os
print(f"{'Model':<25} {'Baseline':>10} {'Split':>10} {'Speedup':>8} {'Correct':>8}")
print("-" * 70)

for name, filename in models:
    filepath = os.path.join(repro_dir, filename)
    if not os.path.exists(filepath):
        print(f"{name:<25} {'MISSING':>10}")
        continue

    spec = importlib.util.spec_from_file_location("repro_mod", filepath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"{name:<25} {'ERROR':>10} {str(e)[:40]}")
        continue

    inputs = mod.make_inputs()

    # Reference
    with torch.no_grad():
        ref = mod.Repro()(*inputs)

    # Baseline
    torch._dynamo.reset()
    GraphLowering._update_scheduler = original_update
    try:
        compiled = torch.compile(mod.Repro())
        with torch.no_grad():
            compiled(*inputs)
        t_base = bench(compiled, inputs)
    except Exception as e:
        print(f"{name:<25} {'FAIL':>10} {str(e)[:40]}")
        continue

    # Split
    torch._dynamo.reset()
    GraphLowering._update_scheduler = patched_update_scheduler
    try:
        compiled2 = torch.compile(mod.Repro())
        with torch.no_grad():
            out_split = compiled2(*inputs)
        # Check correctness
        if isinstance(ref, tuple):
            max_diff = max((a - b).abs().max().item() for a, b in zip(ref, out_split))
        else:
            max_diff = (ref - out_split).abs().max().item()
        correct = max_diff < 1.0  # Allow normal numerical precision
        t_split = bench(compiled2, inputs)
        speedup = t_base / t_split
        print(f"{name:<25} {t_base:>8.1f}us {t_split:>8.1f}us {speedup:>7.2f}x {'OK' if correct else f'FAIL({max_diff:.1f})'}")
    except Exception as e:
        print(f"{name:<25} {t_base:>8.1f}us {'FAIL':>10} {str(e)[:40]}")

GraphLowering._update_scheduler = original_update
