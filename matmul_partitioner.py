"""Select MM benchmark regions from a full post-grad FX graph.

Prologue/epilogue use unchanged normal BB partitions (mm_partition_absorption):
1. A partition absorbs one available producer mm/addmm/bmm (epilogue).
2. A still-unused partition with no scan and at most one reduction, which must
   normalize rows of the MM's A operand along K, absorbs one available
   consumer mm/addmm/bmm through a multiplied operand (prologue).
3. Every original partition/MM is used once; augmented partitions do not grow.
4. Only augmented partitions containing exactly one MM are extracted.
Each is then trimmed to the outputs that depend on its MM and kept only if
0 < non-MM compute <= MatmulLimits.max_normal_compute and it reads no CPU
tensor; epilogues also read at most max_epilogue_extra_inputs extra tensors.
Their category is the source absorption direction.

Horizontal/horizontal_combine keep the independent 2..3-MM window search:
a shared operand (same slot and values, operand_keys), equal output shapes,
and optional surrounding local work within MatmulLimits. These windows may
overlap the normal-partition examples; the model graph is never rewritten.

Every region returns values, not views of them (output_values): an output that
views a value whole is that value, and outputs of one value are one output.
"""

from __future__ import annotations

import operator
from collections import deque
from dataclasses import dataclass
import torch
from torch.fx import Node
from torch.utils._pytree import tree_leaves

MM_NAMES = frozenset(
    {
        "aten::mm",
        "aten::bmm",
        "aten::matmul",
        "aten::addmm",
        "aten::baddbmm",
        "aten::mv",
        "aten::addmv",
        "aten::_scaled_mm",
        "aten::_int_mm",
        "aten::_grouped_mm",
        "aten::_scaled_grouped_mm",
    }
)
INDEX_NAMES = frozenset(
    {
        "aten::index",
        "aten::index_select",
        "aten::gather",
        "aten::embedding",
        "aten::index_put",
        "aten::index_put_",
        "aten::index_copy",
        "aten::index_copy_",
        "aten::scatter",
        "aten::scatter_",
        "aten::scatter_add",
        "aten::scatter_add_",
        "aten::scatter_reduce",
        "aten::scatter_reduce_",
    }
)
VIEW_NAMES = frozenset(
    {
        "aten::view",
        "aten::reshape",
        "aten::_unsafe_view",
        "aten::permute",
        "aten::t",
        "aten::transpose",
        "aten::expand",
        "aten::slice",
        "aten::select",
        "aten::squeeze",
        "aten::unsqueeze",
        "aten::as_strided",
        "aten::split",
        "aten::split_with_sizes",
        "aten::unbind",
        "aten::alias",
        "aten::detach",
    }
)
# Ops that only reinterpret, cast, copy or allocate data, or query sizes.
# They can be inside a region but never count as real computation.
SCAFFOLD_NAMES = VIEW_NAMES | frozenset(
    {
        "aten::_to_copy",
        "prims::convert_element_type",
        "aten::clone",
        "aten::contiguous",
        "aten::copy",
        "aten::copy_",
        "aten::empty",
        "aten::empty_strided",
        "aten::empty_like",
        "aten::zeros",
        "aten::zeros_like",
        "aten::ones",
        "aten::ones_like",
        "aten::full",
        "aten::full_like",
        "aten::lift_fresh_copy",
        "aten::size",
        "aten::sym_size",
        "aten::stride",
        "aten::sym_stride",
        "aten::numel",
        "aten::sym_numel",
    }
)
# Ops that read only the shape/dtype of their first argument, not its values.
SHAPE_ONLY_NAMES = frozenset(
    {
        "aten::empty_like",
        "aten::zeros_like",
        "aten::ones_like",
        "aten::full_like",
        "aten::rand_like",
        "aten::randn_like",
        "aten::size",
        "aten::sym_size",
        "aten::stride",
        "aten::sym_stride",
        "aten::numel",
        "aten::sym_numel",
    }
)


def op_name(node):
    """Schema name such as "aten::mm", or "" for targets without a schema."""
    return getattr(getattr(node.target, "_schema", None), "name", "")


def is_matmul(node):
    return node.op == "call_function" and op_name(node) in MM_NAMES


def is_reduction(node):
    """MM-family ops and ops tagged torch.Tag.reduction."""
    return is_matmul(node) or (
        node.op == "call_function"
        and torch.Tag.reduction in getattr(node.target, "tags", ())
    )


# Ops where one output element combines a whole axis, window or segment of an
# input. Elementwise and gather ops instead read a few computable positions:
#   elementwise a[i,k] = x[i,k] * 2
#   gather      a[i,k] = table[ids[i], k]
#   reduction   s[i]   = x[i,0] + x[i,1] + ... + x[i,K-1]     (sum, var_mean, norms, losses)
#   scan        c[i,k] = x[i,0] + ... + x[i,k]                  (cumsum, nonzero, unique)
#   sort        y[i,:] = sorted(x[i,:])                         (sort, topk, median)
#   window      y[i,j] = max(x[i,2j], x[i,2j+1])                (pooling)
#   accumulate  y[ids[k], j] += x[k, j]                         (index_add, *_backward)
# Tags alone cannot find them: Tag.reduction is missing on aten.cumsum,
# aten._softmax, aten.native_layer_norm and prims.sum, while Tag.pointwise is
# missing on gathers (aten.embedding, aten.index.Tensor), constants (aten.full,
# prims.iota) and views (aten.split, aten._unsafe_view). Matmuls and convolutions
# are not listed: normal BB partitions never contain them.
REDUCTION_SCAN_NAMES = frozenset(
    {
        # Reductions over one or more axes.
        "aten::all", "aten::any", "aten::amax", "aten::amin", "aten::aminmax",
        "aten::_aminmax", "aten::argmax", "aten::argmin", "aten::count_nonzero",
        "aten::max", "aten::min", "aten::mean", "aten::nanmean", "aten::sum",
        "aten::nansum", "aten::sum_to_size", "aten::prod", "aten::std",
        "aten::std_mean", "aten::var", "aten::var_mean", "aten::logsumexp",
        "aten::special_logsumexp", "aten::trace", "aten::_sparse_sum",
        "aten::_sparse_sum_backward", "aten::_sparse_csr_sum", "aten::_sparse_csr_prod",
        "aten::_foreach_max", "aten::_foreach_norm", "aten::_foreach_powsum",
        "aten::segment_reduce", "aten::_segment_reduce_backward",
        "aten::_sparse_mm_reduce_impl", "aten::_sparse_mm_reduce_impl_backward",
        "aten::bincount", "aten::histc", "aten::histogram", "aten::histogramdd",
        "aten::_histogramdd_bin_edges", "aten::_histogramdd_from_bin_cts",
        "aten::_histogramdd_from_bin_tensors", "aten::trapezoid", "aten::trapz",
        "prims::sum", "prims::prod", "prims::xor_sum", "prims::amax",
        "prims::amin", "prims::var", "prims::prepare_softmax_online",
        # Norms and distances.
        "aten::norm", "aten::native_norm", "aten::norm_except_dim",
        "aten::linalg_vector_norm", "aten::linalg_matrix_norm", "aten::linalg_norm",
        "aten::linalg__powsum", "aten::frobenius_norm", "aten::nuclear_norm",
        "aten::renorm", "aten::renorm_", "aten::embedding_renorm",
        "aten::embedding_renorm_", "aten::_weight_norm", "aten::_weight_norm_interface",
        "aten::_weight_norm_interface_backward",
        "aten::_weight_norm_differentiable_backward", "aten::dist", "aten::cdist",
        "aten::_cdist_forward", "aten::_cdist_backward", "aten::_euclidean_dist",
        "aten::pdist", "aten::_pdist_forward", "aten::_pdist_backward",
        "aten::pairwise_distance",
        # Softmax.
        "aten::softmax", "aten::_softmax", "aten::special_softmax",
        "aten::log_softmax", "aten::_log_softmax", "aten::special_log_softmax",
        "aten::_safe_softmax", "aten::_masked_softmax", "aten::_sparse_softmax",
        "aten::_sparse_log_softmax", "aten::_nested_tensor_softmax_with_shape",
        "aten::_softmax_backward_data", "aten::_log_softmax_backward_data",
        "aten::_masked_softmax_backward", "aten::_sparse_softmax_backward_data",
        "aten::_sparse_log_softmax_backward_data",
        # Normalizations that compute statistics.
        "aten::layer_norm", "aten::native_layer_norm",
        "aten::native_layer_norm_backward", "aten::group_norm",
        "aten::native_group_norm", "aten::native_group_norm_backward",
        "aten::instance_norm", "aten::rms_norm", "aten::_fused_rms_norm",
        "aten::_fused_rms_norm_backward", "aten::batch_norm",
        "aten::native_batch_norm", "aten::native_batch_norm_backward",
        "aten::_native_batch_norm_legit", "aten::_native_batch_norm_legit_functional",
        "aten::_batch_norm_with_update", "aten::_batch_norm_with_update_functional",
        "aten::_batch_norm_impl_index", "aten::_batch_norm_impl_index_backward",
        "aten::batch_norm_backward", "aten::batch_norm_stats",
        "aten::batch_norm_gather_stats", "aten::batch_norm_gather_stats_with_counts",
        "aten::batch_norm_backward_reduce", "aten::batch_norm_update_stats",
        "aten::cudnn_batch_norm", "aten::cudnn_batch_norm_backward",
        "aten::miopen_batch_norm", "aten::miopen_batch_norm_backward",
        # Losses that reduce over the batch or classes.
        "aten::binary_cross_entropy", "aten::binary_cross_entropy_with_logits",
        "aten::cosine_embedding_loss", "aten::cross_entropy_loss", "aten::ctc_loss",
        "aten::_ctc_loss", "aten::_ctc_loss_backward", "aten::_cudnn_ctc_loss",
        "aten::miopen_ctc_loss", "aten::hinge_embedding_loss", "aten::huber_loss",
        "aten::kl_div", "aten::l1_loss", "aten::margin_ranking_loss",
        "aten::mse_loss", "aten::multi_margin_loss", "aten::multi_margin_loss_backward",
        "aten::multilabel_margin_loss", "aten::multilabel_margin_loss_forward",
        "aten::multilabel_margin_loss_backward", "aten::nll_loss", "aten::nll_loss_nd",
        "aten::nll_loss_forward", "aten::nll_loss2d", "aten::nll_loss2d_forward",
        "aten::poisson_nll_loss", "aten::smooth_l1_loss", "aten::soft_margin_loss",
        "aten::triplet_margin_loss",
        # Attention (contains softmax).
        "aten::scaled_dot_product_attention",
        "aten::_scaled_dot_product_attention_math",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_flash_attention_backward",
        "aten::_scaled_dot_product_flash_attention_for_cpu",
        "aten::_scaled_dot_product_flash_attention_for_cpu_backward",
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_efficient_attention_backward",
        "aten::_scaled_dot_product_cudnn_attention",
        "aten::_scaled_dot_product_cudnn_attention_backward",
        "aten::_scaled_dot_product_fused_attention_overrideable",
        "aten::_scaled_dot_product_fused_attention_overrideable_backward",
        "aten::_flash_attention_forward", "aten::_flash_attention_backward",
        "aten::_efficient_attention_forward", "aten::_efficient_attention_backward",
        # Scans, including prefix-sum compaction.
        "aten::cumsum", "aten::cumsum_", "aten::cumprod", "aten::cumprod_",
        "aten::cumprod_backward", "aten::cummax", "aten::cummin",
        "aten::_cummax_helper", "aten::_cummin_helper", "aten::cummaxmin_backward",
        "aten::logcumsumexp", "aten::_logcumsumexp", "aten::cumulative_trapezoid",
        "aten::masked_scatter", "aten::masked_scatter_", "aten::masked_scatter_backward",
        "aten::masked_select", "aten::masked_select_backward", "aten::nonzero",
        "aten::nonzero_static", "aten::nonzero_numpy", "aten::argwhere",
        "aten::_unique", "aten::_unique2", "aten::unique_dim",
        "aten::unique_consecutive", "aten::unique_dim_consecutive",
        # Sorts and order statistics.
        "aten::sort", "aten::argsort", "aten::msort", "aten::topk",
        "aten::kthvalue", "aten::median", "aten::nanmedian", "aten::mode",
        "aten::quantile", "aten::nanquantile",
        # Pooling (window reductions) and their backwards.
        "aten::avg_pool1d", "aten::avg_pool2d", "aten::avg_pool3d",
        "aten::avg_pool2d_backward", "aten::avg_pool3d_backward",
        "aten::adaptive_avg_pool1d", "aten::adaptive_avg_pool2d",
        "aten::adaptive_avg_pool3d", "aten::_adaptive_avg_pool2d",
        "aten::_adaptive_avg_pool3d", "aten::_adaptive_avg_pool2d_backward",
        "aten::_adaptive_avg_pool3d_backward", "aten::adaptive_avg_pool3d_backward",
        "aten::adaptive_max_pool1d", "aten::adaptive_max_pool2d",
        "aten::adaptive_max_pool3d", "aten::adaptive_max_pool2d_backward",
        "aten::adaptive_max_pool3d_backward", "aten::max_pool1d",
        "aten::max_pool1d_with_indices", "aten::max_pool2d",
        "aten::max_pool2d_backward", "aten::max_pool2d_with_indices",
        "aten::max_pool2d_with_indices_backward", "aten::max_pool3d",
        "aten::max_pool3d_with_indices", "aten::max_pool3d_with_indices_backward",
        "aten::fractional_max_pool2d", "aten::fractional_max_pool2d_backward",
        "aten::fractional_max_pool3d", "aten::fractional_max_pool3d_backward",
        "aten::mkldnn_max_pool2d", "aten::mkldnn_max_pool2d_backward",
        "aten::mkldnn_max_pool3d", "aten::mkldnn_max_pool3d_backward",
        "aten::mkldnn_adaptive_avg_pool2d", "aten::mkldnn_adaptive_avg_pool2d_backward",
        "aten::quantized_max_pool1d", "aten::quantized_max_pool2d",
        "aten::quantized_max_pool3d", "prims::_low_memory_max_pool_with_offsets",
        # Scatter-accumulation: several inputs are added into one output element.
        "aten::index_add", "aten::index_add_", "aten::index_reduce",
        "aten::index_reduce_", "aten::scatter_add", "aten::scatter_add_",
        "aten::scatter_reduce", "aten::scatter_reduce_",
        "aten::_unsafe_masked_index_put_accumulate", "aten::embedding_bag",
        "aten::_embedding_bag", "aten::_embedding_bag_forward_only",
        "aten::embedding_backward", "aten::embedding_dense_backward",
        "aten::embedding_sparse_backward", "aten::_embedding_bag_backward",
        "aten::_embedding_bag_dense_backward", "aten::_embedding_bag_sparse_backward",
        "aten::_embedding_bag_per_sample_weights_backward", "aten::gather_backward",
        "aten::_gather_sparse_backward", "aten::index_select_backward",
        "aten::col2im", "aten::unfold_backward", "aten::upsample_nearest1d_backward",
        "aten::upsample_nearest2d_backward", "aten::upsample_nearest3d_backward",
        "aten::_upsample_nearest_exact1d_backward",
        "aten::_upsample_nearest_exact2d_backward",
        "aten::_upsample_nearest_exact3d_backward",
        "aten::upsample_linear1d_backward", "aten::upsample_bilinear2d_backward",
        "aten::upsample_bicubic2d_backward", "aten::upsample_trilinear3d_backward",
        "aten::_upsample_bilinear2d_aa_backward", "aten::_upsample_bicubic2d_aa_backward",
        "aten::_upsample_lanczos2d_aa_backward", "aten::grid_sampler_2d_backward",
        "aten::grid_sampler_3d_backward", "aten::_grid_sampler_2d_cpu_fallback_backward",
        "aten::cudnn_grid_sampler_backward", "aten::reflection_pad1d_backward",
        "aten::reflection_pad2d_backward", "aten::reflection_pad3d_backward",
        "aten::replication_pad1d_backward", "aten::replication_pad2d_backward",
        "aten::replication_pad3d_backward",
    }
)

# These accumulate only when asked: index_put(..., accumulate=True) adds into
# y[ids[k]], while accumulate=False writes; scatter accumulates only with reduce=.
_ACCUMULATE_FLAG_NAMES = frozenset(
    {
        "aten::index_put", "aten::index_put_", "aten::_index_put_impl",
        "aten::_index_put_impl_", "aten::_unsafe_index_put",
        "prims::_unsafe_index_put_", "aten::put", "aten::put_",
    }
)
_SCATTER_NAMES = frozenset({"aten::scatter", "aten::scatter_"})
_SCAN_HIGHER_ORDER_OPS = frozenset({"scan", "associative_scan"})


def _schema_argument(node, name):
    """Value of schema argument ``name`` for a call, or None if the op has none."""
    for i, argument in enumerate(node.target._schema.arguments):
        if argument.name == name:
            if i < len(node.args) and not argument.kwarg_only:
                return node.args[i]
            return node.kwargs.get(name, argument.default_value)
    return None


def is_reduction_or_scan(node):
    """Whether one output element of ``node`` combines a whole axis, window or segment.

    The pointwise overloads of a listed op, such as aten.max.other, do not count.
    """
    if node.op != "call_function":
        return False
    target = node.target
    if isinstance(target, torch._ops.HigherOrderOperator):
        return target.name() in _SCAN_HIGHER_ORDER_OPS
    if not isinstance(target, torch._ops.OpOverload):
        return False
    if torch.Tag.pointwise in target.tags:
        return False
    if torch.Tag.reduction in target.tags:
        return True
    name = op_name(node)
    if name in REDUCTION_SCAN_NAMES:
        return True
    if name in _ACCUMULATE_FLAG_NAMES:
        return bool(_schema_argument(node, "accumulate"))
    if name in _SCATTER_NAMES:
        return _schema_argument(node, "reduce") is not None
    return False


def tensor_shape(node):
    """Concrete shape of the node's tensor (first tensor of a tuple), or None."""
    value = node.meta.get("val")
    if isinstance(value, (tuple, list)):
        value = next((x for x in value if isinstance(x, torch.Tensor)), None)
    if isinstance(value, torch.Tensor):
        dims = value.shape
    else:
        meta = node.meta.get("tensor_meta")
        if meta is None or not hasattr(meta, "shape"):
            return None
        dims = meta.shape
    result = []
    for dim in dims:
        if isinstance(dim, int):
            result.append(dim)
        elif isinstance(dim, torch.SymInt) and dim.node.has_hint():
            result.append(dim.node.hint)
        else:
            return None
    return tuple(result)


def value_inputs(node):
    """Inputs whose values the node reads.

    zeros_like/size/... read only metadata of their first argument, and copy_
    overwrites its destination, so those arguments are not value inputs.
    """
    name = op_name(node)
    if name in SHAPE_ONLY_NAMES:
        reference = node.args[0] if node.args else node.kwargs.get("self")
        return [n for n in node.all_input_nodes if n is not reference]
    if name == "aten::copy_":
        return [n for n in node.all_input_nodes if n is not node.args[0]]
    return node.all_input_nodes


def is_mutating(node):
    schema = getattr(node.target, "_schema", None)
    return bool(schema and schema.is_mutable)


def _neutral(node):
    """Identity arithmetic such as x * 1, x + 0, x - 0, x / 1 or x ** 1."""
    if len(node.args) < 2:
        return False
    a, b = node.args[:2]
    scalar = lambda x, v: type(x) in (int, float) and x == v
    name = op_name(node)
    if name in {"aten::mul", "prims::mul"}:
        return scalar(a, 1) or scalar(b, 1)
    if name in {"aten::add", "prims::add"}:
        return node.kwargs.get("alpha", 1) == 1 and (scalar(a, 0) or scalar(b, 0))
    if name in {"aten::sub", "prims::sub"}:
        return scalar(b, 0) and node.kwargs.get("alpha", 1) == 1
    if name in {"aten::div", "prims::div", "aten::pow", "prims::pow"}:
        return scalar(b, 1) and node.kwargs.get("rounding_mode") is None
    return False


def is_computation(node):
    """Real non-MM work: not scaffolding, a tuple projection or identity math."""
    return (
        node.op == "call_function"
        and not is_matmul(node)
        and node.target is not operator.getitem
        and op_name(node) not in SCAFFOLD_NAMES
        and tensor_shape(node) is not None
        and not _neutral(node)
    )


def cpu_inputs(nodes):
    """CPU tensors from outside ``nodes`` that a region with GPU MMs reads.

    DeBERTa-v2 scales attention scores by a 0-d CPU tensor, so the epilogue of
    its key projection reads

        div = torch.ops.aten.div.Tensor(permute_1, arg3_1)   # arg3_1: bf16[] on cpu

    A GPU benchmark of such a region also measures getting a host value into
    the kernel, which is not MM fusion, so regions with CPU inputs are left out.
    Constants a region builds itself, such as full([], 8.0, device=cpu), are
    not inputs and stay.
    """
    def on_cpu(node):
        value = node.meta.get("val")
        return isinstance(value, torch.Tensor) and value.device.type == "cpu"

    if all(on_cpu(n) for n in nodes if is_matmul(n)):
        return set()  # a CPU graph (unit tests) has nothing to mix
    core = set(nodes)
    return {a for n in nodes for a in n.all_input_nodes if a not in core and on_cpu(a)}


_MM_WITH_BIAS = {"aten::addmm", "aten::baddbmm", "aten::addmv"}


def mm_operands(node):
    """The two multiplied operands of an MM (skipping the bias of addmm & co)."""
    start = 1 if op_name(node) in _MM_WITH_BIAS else 0
    return [a for a in node.args[start : start + 2] if isinstance(a, Node)]


class _ReplaySafety:
    """Can a node subset of the graph run on its own, in one piece?

    Uses value dependencies and the alias/write sets of op schemas: no value
    may leave the subset and flow back into it, and no outside op may write
    memory the subset uses while it runs.
    """

    def __init__(self, gm):
        self.nodes = tuple(gm.graph.nodes)
        self.order = {n: i for i, n in enumerate(self.nodes)}
        self.shapes = {n: tensor_shape(n) for n in self.nodes}
        # aliases[n]: memory n's value may share: n itself, its storage, and
        # everything the base of a view or in-place op may alias.
        self.aliases = {}
        for node in self.nodes:
            keys = {("node", node)}
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor) and value.layout == torch.strided:
                keys.add(("storage", value.untyped_storage()._cdata))
            if (
                op_name(node) in VIEW_NAMES
                and node.args
                and isinstance(node.args[0], Node)
            ):
                keys.update(self.aliases[node.args[0]])
            if is_mutating(node) and node.args and isinstance(node.args[0], Node):
                keys.update(self.aliases[node.args[0]])
            self.aliases[node] = keys
        # Memory each node reads, and memory its schema says it writes.
        self.reads = {
            n: set().union(*(self.aliases[a] for a in n.all_input_nodes))
            for n in self.nodes
        }
        self.writes = {}
        for node in self.nodes:
            schema = getattr(node.target, "_schema", None)
            keys = set()
            if schema is not None:
                for i, arg in enumerate(schema.arguments):
                    if arg.alias_info is not None and arg.alias_info.is_write:
                        value = (
                            node.args[i]
                            if i < len(node.args)
                            else node.kwargs.get(arg.name)
                        )
                        for item in tree_leaves(value):
                            if isinstance(item, Node):
                                keys.update(self.aliases[item])
            self.writes[node] = keys

    def safe(self, core):
        last = max(self.order[n] for n in core)
        # (a) No value leaves the candidate and flows back into it.
        pending = [
            u for n in core for u in n.users if u not in core and self.order[u] < last
        ]
        seen = set()
        while pending:
            n = pending.pop()
            if n in seen:
                continue
            seen.add(n)
            if any(u in core for u in n.users):
                return False
            pending.extend(u for u in n.users if self.order[u] < last)
        # (b) No outside op between the candidate's first and last node writes
        #     memory the candidate reads or writes.
        reads, writes = self.reads, self.writes
        ordered = sorted(core, key=self.order.__getitem__)
        touched = set().union(*(reads[n] | writes[n] for n in core))
        if any(
            writes[n] & touched
            for n in self.nodes[self.order[ordered[0]] : last]
            if n not in core
        ):
            return False
        # (c) Inside, reading memory that an earlier node wrote requires a
        #     data dependency on that writer, so replay keeps the order.
        reach = {}
        for n in ordered:
            reach[n] = set().union(
                *(reach[a] | {a} for a in value_inputs(n) if a in core)
            )
            if any(
                writes[p] & reads[n] and p not in reach[n]
                for p in ordered
                if self.order[p] < self.order[n]
            ):
                return False
        return True


CATEGORIES = (
    "prologue",
    "epilogue",
    "horizontal",
    "horizontal_combine",
)

# What shared_operand() looks through between an MM operand and an earlier
# node. Copies keep the logical values; casts, reshapes, expands and permutes
# change the dtype or the index mapping and are recorded as steps.
_COPIES = {
    "aten::clone",
    "aten::contiguous",
    "aten::alias",
    "aten::detach",
    "aten::lift_fresh_copy",
}
_CASTS = {"aten::_to_copy", "prims::convert_element_type"}
_RESHAPES = {
    "aten::view",
    "aten::reshape",
    "aten::_unsafe_view",
    "aten::squeeze",
    "aten::unsqueeze",
    "aten::flatten",
}
_PERMUTES = {"aten::permute", "aten::t", "aten::transpose"}


def _shape_key(node):
    """Hashable shape of the node's tensor; symbolic sizes by expression."""
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    return tuple(d if isinstance(d, int) else str(d) for d in value.shape)


def _permutation(node, rank):
    """permute/t/transpose as dims: output dim i is input dim perm[i]."""
    perm = list(range(rank))
    name = op_name(node)
    if name == "aten::permute":
        return tuple(d % rank for d in _schema_argument(node, "dims"))
    if name == "aten::t" or rank == 0:
        return tuple(reversed(perm))  # t() is the identity on 0-D and 1-D tensors
    d0, d1 = (_schema_argument(node, d) % rank for d in ("dim0", "dim1"))
    perm[d0], perm[d1] = perm[d1], perm[d0]
    return tuple(perm)


def _merge_steps(steps):
    merged = []
    for kind, arg in steps:
        if merged and kind == merged[-1][0] == "reshape":
            merged[-1] = (kind, arg)  # reshape(reshape(x, a), b) == reshape(x, b)
        elif merged and kind == merged[-1][0] == "permute":
            first = merged.pop()[1]
            merged.append((kind, tuple(first[d] for d in arg)))
        else:
            merged.append((kind, arg))
        if merged[-1] == ("permute", tuple(range(len(merged[-1][1])))):
            merged.pop()  # identity, e.g. x.t().t()
    return tuple(merged)


def shared_operand(value, within=None):
    """(root, steps) such that ``value`` is ``steps`` applied to node ``root``.

    Operands with equal (root, steps) hold equal values, so they can be the
    shared operand of one concatenated GEMM:
      x @ a,        x @ b              (x, ())        == (x, ())
      x @ a,        x.t() @ b          (x, ())        != (x, (permute (1, 0),))
      x.half() @ a, x.bfloat16() @ b   (x, (cast f16,)) != (x, (cast bf16,))
    Copies are dropped and runs of reshapes or permutes are merged, so two
    separate x.half() or x.view(m, k) nodes match. With ``within`` (a node
    set) the walk stops at nodes outside it: once the set is extracted, they
    are inputs.
    """
    steps = []
    while (
        isinstance(value, Node)
        and (within is None or value in within)
        and value.args
        and isinstance(value.args[0], Node)
    ):
        name, source = op_name(value), value.args[0]
        if name in _COPIES:
            pass
        elif name in _CASTS:
            out, inp = value.meta.get("val"), source.meta.get("val")
            if not (isinstance(out, torch.Tensor) and isinstance(inp, torch.Tensor)):
                break
            if (out.dtype, out.device) != (inp.dtype, inp.device):
                steps.append(("cast", (out.dtype, out.device)))
        elif name in _RESHAPES or name == "aten::expand":
            shape = _shape_key(value)
            if shape is None:
                break
            steps.append(("reshape" if name in _RESHAPES else "expand", shape))
        elif name in _PERMUTES:
            source_value = source.meta.get("val")
            if not isinstance(source_value, torch.Tensor):
                break
            steps.append(("permute", _permutation(value, source_value.dim())))
        else:
            break
        value = source
    return value, _merge_steps(reversed(steps))


def operand_keys(mm, within=None):
    """(slot, root, steps) of each multiplied operand; slot 0 is the left one.

    x @ a and x @ b share (0, x, ()) and concatenate along N; a @ x and b @ x
    share (1, x, ()) and concatenate along M. x @ a and b @ x share nothing.
    """
    start = 1 if op_name(mm) in _MM_WITH_BIAS else 0
    return {
        (slot, *shared_operand(operand, within))
        for slot, operand in enumerate(mm.args[start : start + 2])
        if isinstance(operand, Node)
    }


def shares_operand(mms, within=None):
    """All of ``mms`` have a multiplied operand with the same operand_keys key."""
    return bool(set.intersection(*(operand_keys(mm, within) for mm in mms)))


def _operand_at(mm, slot):
    return mm.args[(1 if op_name(mm) in _MM_WITH_BIAS else 0) + slot]


def _operand_path(value, root):
    """The nodes shared_operand() walked from ``value`` to ``root``, excluding root."""
    path = []
    while value is not root:
        path.append(value)
        value = value.args[0]
    return path


@dataclass(frozen=True)
class MatmulLimits:
    """Size limits of extracted regions (also applied by curation).

    max_horizontal..max_outputs bound horizontal/horizontal_combine windows.
    Prologue/epilogue partitions are whole BB partitions, trimmed to the
    outputs that depend on their MM; max_normal_compute bounds both and
    max_epilogue_extra_inputs bounds epilogues.
    """

    max_horizontal: int = 3  # MMs in a horizontal group; growth stops at 3
    max_mm: int = 3
    max_nodes: int = 48
    max_compute: int = 32  # MMs plus real computation
    max_inputs: int = 12
    max_outputs: int = 4
    # Prologue/epilogue keep 0 < non-MM compute <= this, after trimming:
    # 0 is a bare MM (views/casts only), larger is mostly unrelated work.
    max_normal_compute: int = 31
    # Epilogues read at most this many extra tensors: inputs other than the
    # MM operands (A, B, addmm bias) and the dropout seed buffer.
    max_epilogue_extra_inputs: int = 4


class MatmulRegion(list):
    """The FX nodes of one region, in graph order.

    outputs: nodes whose values the benchmark returns (output_values).
    metrics: sizes of the region.
    category: one of CATEGORIES.
    mm_types: sorted MM op names such as ("addmm", "bmm"), used for naming.
    """

    def __init__(self, nodes, outputs, metrics, category):
        super().__init__(nodes)
        self.outputs = tuple(outputs)
        self.metrics = metrics
        self.category = category
        self.metrics["category"] = category
        self.mm_types = tuple(
            sorted({op_name(n).split("::")[-1] for n in nodes if is_matmul(n)})
        )


# ---------------------------------------------------------------------------
# Region search
# ---------------------------------------------------------------------------


class _Extractor:
    def __init__(self, gm, supported, limits):
        self.limits = limits
        self.supported = supported
        self.collector = _ReplaySafety(gm)
        self.nodes = self.collector.nodes
        self.order = self.collector.order
        self.shapes = self.collector.shapes
        self.rejections = {}
        # mm_dependent[n]: n depends on a supported MM through supported nodes.
        self.mm_dependent = {}
        # rooted[n]: n is pointwise/transparent work computed only from graph
        # inputs and constants (a dropout mask, an iota mask, gelu'(saved x)).
        self.rooted = {}
        for node in self.nodes:
            self.mm_dependent[node] = node in supported and (
                is_matmul(node) or any(self.mm_dependent[a] for a in value_inputs(node))
            )
            self.rooted[node] = (
                node in supported
                and not is_reduction(node)
                and node.target is not operator.getitem
                and all(
                    a.op in {"placeholder", "get_attr"} or self.rooted[a]
                    for a in value_inputs(node)
                )
            )

    def reject(self, reason):
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def sort(self, nodes):
        return sorted(nodes, key=self.order.__getitem__)

    def connector(self, node, core):
        """Pointwise, indexing or transparent node that may join a region."""
        if node not in self.supported or is_reduction(node):
            return False
        if node.target is operator.getitem:
            source = node.args[0]
            return isinstance(source, Node) and (
                source in core or self.connector(source, core)
            )
        return True

    def metrics(self, core):
        dots = [n for n in core if is_matmul(n)]
        depth = {}
        for n in self.sort(core):
            depth[n] = max(
                (depth[a] for a in value_inputs(n) if a in core), default=0
            ) + int(is_reduction(n))
        return {
            "nodes": len(core),
            "compute": sum(is_matmul(n) or is_computation(n) for n in core),
            "mm": len(dots),
            "inputs": len({a for n in core for a in n.all_input_nodes if a not in core}),
            "contraction_depth": max(depth.values(), default=0),
        }

    def fits(self, core):
        m = self.metrics(core)
        l = self.limits
        return (
            m["nodes"] <= l.max_nodes
            and m["compute"] <= l.max_compute
            and m["mm"] <= l.max_mm
            and m["inputs"] <= l.max_inputs
        )

    def grow_backward(self, core, roots, *, side_branch=False):
        """Absorb pointwise/gather/transparent producers of ``roots``.

        Reductions, MMs and values computed from an MM stay outside and become
        inputs. For a horizontal window's side branch, only transparent and
        gather ops, and work computed only from graph inputs and constants
        (``rooted``), are absorbed. Computing a side input from other
        intermediate values (e.g. a residual's norm affine) is not work
        around the MM.
        """
        pending = deque(self.sort(r for r in roots if r not in core))
        visited = set()
        while pending:
            node = pending.popleft()
            if node in core or node in visited:
                continue
            visited.add(node)
            if not self.connector(node, core) or self.mm_dependent[node]:
                continue
            if side_branch and not (
                node.target is operator.getitem
                or op_name(node) in SCAFFOLD_NAMES | INDEX_NAMES
                or self.rooted[node]
            ):
                continue
            trial = core | {node}
            if not self.fits(trial) or not self.collector.safe(trial):
                visited.discard(node)  # may fit once another path is absorbed
                continue
            core = trial
            pending.extend(self.sort(value_inputs(node)))
        return core

    def grow_forward(self, core, sources):
        """Yield ever larger windows absorbing consumers of ``sources``."""
        yield core
        pending = deque(self.sort(u for s in sources for u in s.users))
        visited = set()
        while pending:
            node = pending.popleft()
            if node in core or node in visited:
                continue
            visited.add(node)
            if not self.connector(node, core):
                continue
            trial = core | {node}
            if not self.fits(trial) or not self.collector.safe(trial):
                # A join is unsafe until its other branch is in the window;
                # adding that branch's last node re-queues it.
                visited.discard(node)
                continue
            core = trial
            yield core
            pending.extend(self.sort(node.users))

    def side_inputs(self, core, exclude=()):
        """Producers of the window's inputs that may be pulled in as branches."""
        return {
            a
            for n in core
            if n not in exclude
            for a in value_inputs(n)
            if a not in core
        }

    def finish(self, core, internal, required=()):
        """Choose outputs (output_values), prune to what they need and check every limit."""
        outputs = output_values(self.sort(
            n
            for n in core
            if internal(n) and (is_mutating(n) or any(u not in core for u in n.users))
        ))
        if not outputs:
            self.reject("no_output")
            return None
        needed = set(outputs)
        stack = list(outputs)
        while stack:
            for a in value_inputs(stack.pop()):
                if a in core and a not in needed:
                    needed.add(a)
                    stack.append(a)
        if any(r not in needed for r in required):
            self.reject("pruned_required")
            return None
        metrics = self.metrics(needed)
        metrics["outputs"] = len(outputs)
        metrics["non_mm_compute"] = sum(is_computation(n) for n in needed)
        if len(outputs) > self.limits.max_outputs or not self.fits(needed):
            self.reject("limits")
            return None
        if cpu_inputs(needed):
            self.reject("cpu_input")
            return None
        if not self.collector.safe(needed):
            self.reject("cycle_or_alias")
            return None
        return self.sort(needed), outputs, metrics

    # --- horizontal benchmark windows ------------------------------------

    def _after(self, node, mm, window):
        """node depends on mm inside window."""
        stack, seen = [node], set()
        while stack:
            n = stack.pop()
            if n is mm:
                return True
            for a in value_inputs(n):
                if a in window and a not in seen:
                    seen.add(a)
                    stack.append(a)
        return False

    def horizontal_groups(self, mms):
        """Groups of 2..max_horizontal independent MMs sharing an operand.

        Sharing means an equal operand_keys() key: the same slot and the same
        values, so x @ a with b @ x, with x.t() @ b, or x.half() @ a with
        x.bfloat16() @ b never group (none concatenates into one GEMM).
        """
        ancestors = {}
        for n in self.nodes:
            ancestors[n] = set().union(
                *(ancestors[a] | ({a} if is_matmul(a) else set()) for a in n.all_input_nodes)
            )
        by_key = {}
        for mm in mms:
            shape = self.shapes.get(mm)
            if shape is None:
                continue
            for key in operand_keys(mm):
                by_key.setdefault((key, shape), []).append(mm)
        groups, seen = [], set()
        for members in sorted(by_key.values(), key=lambda m: self.order[m[0]]):
            chosen = []
            for mm in self.sort(members):
                if all(mm not in ancestors[c] and c not in ancestors[mm] for c in chosen):
                    chosen.append(mm)
                if len(chosen) == self.limits.max_horizontal:
                    break  # a 4th sharing MM is never added
            key = frozenset(chosen)
            if len(chosen) >= 2 and key not in seen:
                seen.add(key)
                groups.append(chosen)
        return groups

    def horizontal(self, group):
        members = set(group)
        slot, root, _ = min(
            set.intersection(*(operand_keys(mm) for mm in group)),
            key=lambda key: (key[0], self.order.get(key[1], -1)),
        )
        # The operand paths up to where they meet must be in the window, but
        # grow_backward leaves out nodes computed from an earlier MM. Without
        # them each MM reads its own input and the sharing is lost:
        #   x -> view_1 -> mm_1, x -> view_2 -> mm_2   extracts as   mm_1(arg0), mm_2(arg1)
        # Adding view_1 and view_2 makes both read x; one view feeding both
        # MMs is already a single input and stays out.
        paths = [_operand_path(_operand_at(mm, slot), root) + [root] for mm in group]
        common = set.intersection(*map(set, paths))
        path = set()
        for nodes in paths:
            for n in nodes:
                if n in common:
                    break
                path.add(n)
        if not path <= self.supported:
            self.reject("unsupported_shared_operand")
            return None
        core = self.grow_backward(members, [a for mm in group for a in mm_operands(mm)]) | path
        snapshots = list(self.grow_forward(core, group))
        for window in reversed(snapshots):
            window = self.grow_backward(
                window, self.side_inputs(window, members), side_branch=True
            )
            finished = self.finish(
                window, lambda n: any(self._after(n, mm, window) for mm in group), group
            )
            if finished is not None and not shares_operand(group, within=set(finished[0])):
                self.reject("unshared_operand")  # the extracted MMs would read separate inputs
                finished = None
            if finished is not None:
                category = (
                    "horizontal_combine" if combines(finished[0], group) else "horizontal"
                )
                return MatmulRegion(*finished, category)
        return None


def supported_nodes(gm, is_supported, excluded=()):
    """Nodes that may be inside a region at all (standalone-replay boundaries)."""
    excluded = set(excluded)

    def local_copy(node):
        if op_name(node) != "aten::_to_copy" or not node.args or not isinstance(node.args[0], Node):
            return True
        source, result = node.args[0].meta.get("val"), node.meta.get("val")
        return not (
            isinstance(source, torch.Tensor)
            and isinstance(result, torch.Tensor)
            and source.device != result.device
        )

    def tensor_operation(node):
        tags = getattr(node.target, "tags", ())
        return op_name(node).startswith(("aten::", "prims::")) and (
            torch.Tag.pointwise in tags or torch.Tag.reduction in tags
        )

    supported = {
        n
        for n in gm.graph.nodes
        if n not in excluded
        and n.op == "call_function"
        and (
            is_matmul(n)
            or op_name(n) in INDEX_NAMES | SCAFFOLD_NAMES
            or n.target is operator.getitem
            or tensor_operation(n)
            or is_supported(n)
        )
        and local_copy(n)
        and op_name(n) != "aten::unbind"
        and not any(x in op_name(n) for x in ("softmax", "attention"))
    }
    for node in gm.graph.nodes:
        if (
            node in supported
            and node.target is operator.getitem
            and (not isinstance(node.args[0], Node) or node.args[0] not in supported)
        ):
            supported.remove(node)
    return supported


# Views whose result shows every element of their input, unless it is smaller:
# a slice or select only when it covers its whole input (a full slice, a select
# of a size-1 dim).
_WHOLE_VIEWS = _RESHAPES | _PERMUTES | {
    "aten::expand", "aten::alias", "aten::detach", "aten::slice", "aten::select",
}


def view_root(node):
    """``node`` without its trailing views that keep every element of their input."""
    while (
        op_name(node) in _WHOLE_VIEWS
        and node.meta["val"].numel() >= node.args[0].meta["val"].numel()
    ):
        node = node.args[0]
    return node


def output_values(outputs):
    """The values a region returns for its escaping ``outputs``.

    A view returns its input's storage, so an output that views a value whole
    (view_root) returns that value, and outputs that view one value are one
    output. M2M100's addmm + residual + LayerNorm epilogue, which 24 consumers
    read through their own view, was captured as

        convert_element_type_1 = convert_element_type(add_2, torch.bfloat16)
        view_1 = view(convert_element_type_1, _shape_param_1)
        ...
        view_24 = view(convert_element_type_1, _shape_param_24)
        return (convert_element_type_1, view_1, ..., view_24)

    and now ends in ``return convert_element_type_1``, the pattern of the same
    epilogue read through any number of views. A view that drops elements stays
    an output unless it views another output:

        add = add(arg3_1, view)            # [16, 128, 768], BERT_pytorch
        select = select(add, 1, 0)         # [16, 768]
        view_1 = view(add, _shape_param_1)
        return (select, view_1)            # now: return add

    DeiT's ``return (select, select_1)``, its two class tokens of a
    [128, 198, 768] norm, keeps both: returning the norm would claim all 198
    tokens. Casts and clones write a new tensor and are outputs of their own;
    ``return clone(permute, memory_format=torch.contiguous_format)`` is an
    epilogue that stores a transposed copy.

    Callers pass outputs that depend on an MM of the region, so the walks stay
    inside it.
    """
    roots = list(dict.fromkeys(map(view_root, outputs)))
    returned = set(roots)

    def views_another_output(node):
        while op_name(node) in VIEW_NAMES:
            node = node.args[0]
            if node in returned:
                return True
        return False

    return [n for n in roots if not views_another_output(n)]


def trim_to_matmul(nodes, outputs, mm):
    """Keep the outputs that depend on ``mm``, as output_values, and the nodes they need.

    A BB partition can carry work that never touches its absorbed MM (an
    unrelated store fused by the BB partitioner); that is not MM fusion.
    Output views that output_values replaces are dropped unless a kept node
    reads them.
    """
    core = set(nodes)
    after = {mm}
    for node in nodes:  # graph order
        if any(a in after for a in node.all_input_nodes):
            after.add(node)
    kept = output_values([n for n in outputs if n in after])
    needed, stack = set(kept), list(kept)
    while stack:
        for arg in stack.pop().all_input_nodes:
            if arg in core and arg not in needed:
                needed.add(arg)
                stack.append(arg)
    return [n for n in nodes if n in needed], kept


def extra_inputs(nodes, mm):
    """Tensors a region reads besides the MM operands and dropout seeds.

    Operands and seeds are followed back through views/casts inside the
    region, so ``x.view(...) @ w`` still counts x as an operand.
    """
    core = set(nodes)

    def source(node):
        while node in core and op_name(node) in SCAFFOLD_NAMES and node.all_input_nodes:
            node = node.args[0] if isinstance(node.args[0], Node) else node.all_input_nodes[0]
        return node

    skipped = {source(a) for a in mm.all_input_nodes}
    skipped |= {
        source(n.args[0]) for n in nodes
        if op_name(n) == "prims::inductor_lookup_seed" and isinstance(n.args[0], Node)
    }
    return {
        a for n in nodes for a in n.all_input_nodes
        if a not in core and a not in skipped and isinstance(a.meta.get("val"), torch.Tensor)
    }


def _normal_mm_regions(gm, excluded, categories, limits):
    """Augmented normal partitions, trimmed to their MM and bounded in size."""
    from mm_partition_absorption import absorb_matmuls

    regions = []
    for nodes, category in absorb_matmuls(gm, excluded=excluded):
        if category not in categories:
            continue
        core = set(nodes)
        outputs = [n for n in nodes if is_mutating(n) or any(u not in core for u in n.users)]
        (mm,) = [n for n in nodes if is_matmul(n)]
        nodes, outputs = trim_to_matmul(nodes, outputs, mm)
        if not 0 < sum(map(is_computation, nodes)) <= limits.max_normal_compute:
            continue
        extra = len(extra_inputs(nodes, mm))
        if category == "epilogue" and extra > limits.max_epilogue_extra_inputs:
            continue
        if cpu_inputs(nodes):
            continue
        core = set(nodes)
        depth = {}
        for node in nodes:
            depth[node] = max((depth[a] for a in value_inputs(node) if a in core), default=0) + int(is_reduction(node))
        metrics = {
            "nodes": len(nodes), "mm": 1,
            "compute": sum(is_matmul(n) or is_computation(n) for n in nodes),
            "non_mm_compute": sum(is_computation(n) for n in nodes),
            "inputs": len({a for n in nodes for a in n.all_input_nodes if a not in core}),
            "extra_inputs": extra,
            "outputs": len(outputs), "contraction_depth": max(depth.values(), default=0),
            "selection": "normal_absorption_v1",
        }
        regions.append(MatmulRegion(nodes, outputs, metrics, category))
    return regions


def get_matmul_regions(gm, is_supported, *, limits=None, excluded=(), categories=CATEGORIES, diagnostics=None):
    """Return MatmulRegions of ``gm`` for the requested categories."""
    limits = limits or MatmulLimits()
    supported = supported_nodes(gm, is_supported, excluded)
    ex = _Extractor(gm, supported, limits)

    mms = [n for n in ex.nodes if is_matmul(n) and n in supported]
    regions = (
        _normal_mm_regions(gm, excluded, set(categories), limits)
        if {"prologue", "epilogue"} & set(categories) else []
    )
    if {"horizontal", "horizontal_combine"} & set(categories):
        for group in ex.horizontal_groups(mms):
            region = ex.horizontal(group)
            if region is not None and region.category in categories:
                regions.append(region)
    if diagnostics is not None:
        diagnostics.update(
            {
                "mm_seeds": len(mms),
                "regions": {c: sum(r.category == c for r in regions) for c in CATEGORIES},
                "rejections": dict(sorted(ex.rejections.items())),
            }
        )
    return regions


def classify_pattern_graph(gm, *, expected=None):
    """Validate a category; single-MM absorption direction comes from source metadata.

    A single-MM graph needs source metadata to distinguish prologue from
    epilogue. Horizontal graphs require independent MMs; dependent MM chains
    are not a supported category.
    """
    nodes = list(gm.graph.nodes)
    mms = [n for n in nodes if is_matmul(n)]
    reductions = [n for n in nodes if is_reduction(n) and not is_matmul(n)]
    if expected in {"prologue", "epilogue"}:
        if len(mms) != 1 or op_name(mms[0]) not in {"aten::mm", "aten::addmm", "aten::bmm"}:
            return None
        # A prologue may hold one reduction (a K-row norm; absorption checks
        # the axis) and no scan.
        found = [n for n in nodes if is_reduction_or_scan(n)]
        if expected == "prologue" and (len(found) > 1 or not all(map(is_reduction, found))):
            return None
        if not any(n.op == "call_function" and n not in mms for n in nodes):
            return None
        return expected
    if reductions or len(mms) < 2:
        return None
    ancestors = {}
    for n in nodes:
        ancestors[n] = set().union(*(ancestors[a] | {a} for a in value_inputs(n)))
    if any(s in ancestors[t] for s in mms for t in mms):
        return None
    shapes = {tensor_shape(m) for m in mms}
    if len(mms) <= 3 and len(shapes) == 1 and None not in shapes and shares_operand(mms):
        return "horizontal_combine" if combines(nodes, mms) else "horizontal"
    return None


def combines(nodes, mms):
    """Some op of ``nodes`` (graph order) depends on >= 2 of ``mms``.

    The graph's output node does not count: returning several MM results
    separately is not combining them.
    """
    reached = {}
    for n in nodes:
        if n.op != "call_function":
            continue
        reached[n] = set().union(
            *(reached[a] for a in value_inputs(n) if a in reached)
        ) | ({n} if n in mms else set())
        if len(reached[n]) >= 2:
            return True
    return False
