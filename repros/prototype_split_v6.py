"""Prototype v6: Branch-folding inner_fn for each half.

Key: wrap the original inner_fn in a handler that tracks index expression
bounds and folds comparisons when provably true/false.
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

x = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')
sin = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')


def detect_pointwise_cat_split(pw):
    """Detect binary pointwise_cat on innermost dim."""
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


class _TrueConst:
    """Sentinel for a branch condition known to be True."""
    pass

class _FalseConst:
    """Sentinel for a branch condition known to be False."""
    pass

TRUE_CONST = _TrueConst()
FALSE_CONST = _FalseConst()


def make_branch_folding_fn(original_fn, dim_idx, split_point, total_dim, which_half):
    """Create an inner_fn that folds the pointwise_cat branch for one half.
    
    which_half=0: i3 ranges [0, split_point), so i3 < split_point is True
    which_half=1: i3 ranges [0, total-split_point) + offset, so i3+offset < split_point is False
    """
    
    # We intercept ops calls to fold the branch
    # The pattern in the inner_fn is:
    #   tmp = ops.index_expr(idx[dim], int64)  -- the innermost index
    #   cond = ops.lt(tmp, ops.constant(split_point, int64))
    #   val_true = ops.masked(cond, body_true, 0.0)
    #   val_false = ops.masked(not_cond, body_false, 0.0)
    #   result = ops.where(cond, val_true, val_false)
    #
    # For half 0: cond=True, so result=body_true()
    # For half 1: cond=False, so result=body_false()
    
    class BranchFoldingOps:
        """Wraps the real ops handler, folding known-constant branches."""
        
        def __init__(self):
            self._index_expr_values = {}  # track which values came from index_expr of innermost
            self._true_masks = set()  # IDs of conditions known True
            self._false_masks = set()  # IDs of conditions known False
        
        def __getattr__(self, name):
            return getattr(ops, name)
        
        def index_expr(self, expr, dtype):
            result = ops.index_expr(expr, dtype)
            # Track: is this the innermost index expression?
            # For half 0: expr is just i3 (in [0, split_point))
            # For half 1: expr is i3 + split_point (in [split_point, total_dim))
            if isinstance(expr, sympy.Expr):
                # Check if this expr equals the innermost variable (or offset version)
                free = expr.free_symbols
                # We'll use a simpler approach: tag the result
                self._index_expr_values[id(result)] = expr
            return result
        
        def lt(self, a, b):
            # Check if this is comparing our tracked index expr against split_point
            a_expr = self._index_expr_values.get(id(a))
            if a_expr is not None:
                # b should be the constant split_point
                b_val = None
                # b comes from ops.constant or ops.index_expr
                # We stored it... actually let's try a different approach
                pass
            return ops.lt(a, b)
        
        def ge(self, a, b):
            return ops.ge(a, b)
        
        def masked(self, mask, body, other):
            if isinstance(mask, _TrueConst):
                return body()
            if isinstance(mask, _FalseConst):
                return ops.constant(other, torch.float32) if isinstance(other, (int, float)) else other
            return ops.masked(mask, body, other)
        
        def where(self, mask, a, b):
            if isinstance(mask, _TrueConst):
                return a
            if isinstance(mask, _FalseConst):
                return b
            return ops.where(mask, a, b)
    
    # Simpler approach: since we KNOW the structure of pointwise_cat's inner_fn,
    # and we know which half we're in, we can reconstruct a clean inner_fn
    # by calling the original with a special ops interception.
    #
    # But the ops handler approach is complex because we're not calling ops directly
    # (the inner_fn closure calls ops.* methods).
    #
    # SIMPLEST approach: don't fold at IR level. Instead, rely on the fact that
    # when i3 < split_point is ALWAYS true (for half 0), the generated Triton code
    # will have `tl.load(ptr, mask=True)` which Triton handles efficiently.
    # The `tl.where(True, a, b)` is also optimized by Triton.
    #
    # Actually, let's just test: does Triton optimize away the dead branch?
    
    def half_fn(idx):
        new_idx = list(idx)
        if which_half == 1:
            new_idx[dim_idx] = idx[dim_idx] + split_point
        return original_fn(new_idx)
    
    return half_fn


original_update = GraphLowering._update_scheduler

def patched_update_scheduler(self):
    from torch._inductor.scheduler import Scheduler
    
    ops_to_remove = []
    
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
        
        print(f"  Splitting {orig_buf_name}: dim[-1] {split_point}/{total_dim}")
        
        original_fn = pw.inner_fn
        fn_lo = make_branch_folding_fn(original_fn, dim_idx, split_point, total_dim, 0)
        fn_hi = make_branch_folding_fn(original_fn, dim_idx, split_point, total_dim, 1)
        
        ranges_lo = list(pw.ranges)
        ranges_lo[dim_idx] = split_point
        ranges_hi = list(pw.ranges)
        ranges_hi[dim_idx] = second_half
        
        pw_lo = ir.Pointwise(
            device=pw.device, dtype=pw.dtype, inner_fn=fn_lo, ranges=ranges_lo,
        )
        pw_lo._post_init_setattr("origin_node", pw.origin_node)
        pw_lo._post_init_setattr("traceback", pw.traceback)
        
        pw_hi = ir.Pointwise(
            device=pw.device, dtype=pw.dtype, inner_fn=fn_hi, ranges=ranges_hi,
        )
        pw_hi._post_init_setattr("origin_node", pw.origin_node)
        pw_hi._post_init_setattr("traceback", pw.traceback)
        
        sb_lo = ir.StorageBox(pw_lo)
        sb_hi = ir.StorageBox(pw_hi)
        
        output_size = list(pw.ranges)
        output_stride = ir.FlexibleLayout.contiguous_strides(output_size)
        
        concat_kernel = ir.ConcatKernel(
            name=None,
            layout=ir.FixedLayout(
                device=pw.device, dtype=pw.dtype,
                size=output_size, stride=output_stride,
            ),
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
        
        ops_to_remove.append(op)
        print(f"  Sub-buffers: {buf_lo.get_name()}, {buf_hi.get_name()}")
        break
    
    for op in ops_to_remove:
        self.operations.remove(op)
    
    if ops_to_remove:
        print(f"  Operations: {[(type(op).__name__, op.get_name()) for op in self.operations]}")
    
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)

GraphLowering._update_scheduler = patched_update_scheduler

def bench_graph(fn, inputs, warmup=200, iters=1000):
    for _ in range(warmup):
        with torch.no_grad(): fn(*inputs)
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

# Test with split
torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x, cos, sin)
with torch.no_grad():
    ref = LlamaRoPE()(x, cos, sin)
t_split = bench_graph(compiled, [x, cos, sin])
print(f"\nWith split: {t_split:.1f} us, correctness: max_diff={((out - ref).abs().max().item()):.6f}")

# Now let's look at what kernel was generated
import glob
latest = sorted(glob.glob("/tmp/torchinductor_dev/*/??/*.py"), key=lambda f: __import__('os').path.getmtime(f))[-1]
print(f"\nGenerated kernel: {latest}")

# Show the kernel
import subprocess
result = subprocess.run(['grep', '-A3', 'xnumel\|tl.store\|tl.where\|tl.load.*mask\|num_load\|num_store', latest], 
                      capture_output=True, text=True)
print(result.stdout[:2000])
