"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForQuestionAnswering_train_000
Pattern hash: 2a28fce37306
Shape hash: 613155f9
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 2], f32), T([1], i64), T([1], i64), T([], f32), S([1, 128, 2]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_56: "f32[128, 2]", arg314_1: "i64[1]", arg315_1: "i64[1]", full_1: "f32[]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[1, 128, 2]" = torch.ops.aten.view.default(addmm_56, _shape_param_0);  addmm_56 = _shape_param_0 = None
        split_tensor = torch.ops.aten.split.Tensor(view_default, 1, -1);  view_default = None
        getitem: "f32[1, 128, 1]" = split_tensor[0]
        getitem_1: "f32[1, 128, 1]" = split_tensor[1];  split_tensor = None
        squeeze_dim: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem, -1);  getitem = None
        clone_default: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        squeeze_dim_1: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_1, -1);  getitem_1 = None
        clone_default_1: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze_dim_1, memory_format = torch.contiguous_format);  squeeze_dim_1 = None
        clamp_min_default: "i64[1]" = torch.ops.aten.clamp_min.default(arg314_1, 0);  arg314_1 = None
        clamp_max_default: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 128);  clamp_min_default = None
        clamp_min_default_1: "i64[1]" = torch.ops.aten.clamp_min.default(arg315_1, 0);  arg315_1 = None
        clamp_max_default_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_default_1, 128);  clamp_min_default_1 = None
        amax_default: "f32[1, 1]" = torch.ops.aten.amax.default(clone_default, [1], True)
        sub_tensor: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_default, amax_default);  clone_default = amax_default = None
        exp_default: "f32[1, 128]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[1, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_default, 128)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[1]" = torch.ops.aten.where.self(ne_scalar, clamp_max_default, full_default);  clamp_max_default = None
        unsqueeze_default: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[1, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[1]" = torch.ops.aten.neg.default(squeeze_dim_2);  squeeze_dim_2 = None
        where_self_1: "f32[1]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_1);  neg_default = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None
        amax_default_1: "f32[1, 1]" = torch.ops.aten.amax.default(clone_default_1, [1], True)
        sub_tensor_2: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_default_1, amax_default_1);  clone_default_1 = amax_default_1 = None
        exp_default_1: "f32[1, 128]" = torch.ops.aten.exp.default(sub_tensor_2)
        sum_dim_int_list_1: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_1, [1], True);  exp_default_1 = None
        log_default_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_3: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_2, log_default_1);  sub_tensor_2 = log_default_1 = None
        ne_scalar_1: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_default_1, 128)
        where_self_2: "i64[1]" = torch.ops.aten.where.self(ne_scalar_1, clamp_max_default_1, full_default);  clamp_max_default_1 = full_default = None
        unsqueeze_default_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_self_2, 1);  where_self_2 = None
        gather_default_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_tensor_3, 1, unsqueeze_default_1);  sub_tensor_3 = unsqueeze_default_1 = None
        squeeze_dim_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_default_1, 1);  gather_default_1 = None
        neg_default_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_dim_3);  squeeze_dim_3 = None
        where_self_3: "f32[1]" = torch.ops.aten.where.self(ne_scalar_1, neg_default_1, full_1);  neg_default_1 = full_1 = None
        sum_default_2: "i64[]" = torch.ops.aten.sum.default(ne_scalar_1);  ne_scalar_1 = None
        convert_element_type_default_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_2, torch.float32);  sum_default_2 = None
        sum_default_3: "f32[]" = torch.ops.aten.sum.default(where_self_3);  where_self_3 = None
        div_tensor_1: "f32[]" = torch.ops.aten.div.Tensor(sum_default_3, convert_element_type_default_1);  sum_default_3 = convert_element_type_default_1 = None
        add_tensor: "f32[]" = torch.ops.aten.add.Tensor(div_tensor, div_tensor_1);  div_tensor = div_tensor_1 = None
        div_tensor_2: "f32[]" = torch.ops.aten.div.Tensor(add_tensor, 2);  add_tensor = None
        return div_tensor_2

def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)

def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
