"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_infer_000
Pattern hash: 674523e70161
Shape hash: d46ab65d
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 512], i64), T([4096, 30000], f32), S([8, 512, 30000]), S([-1, 30000]))"

class Repro(torch.nn.Module):
    def forward(self, arg31_1: "i64[8, 512]", addmm_74: "f32[4096, 30000]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "i64[4096]" = torch.ops.aten.view.default(arg31_1, [-1]);  arg31_1 = None
        ne_scalar: "b8[4096]" = torch.ops.aten.ne.Scalar(view_default, -100)
        view_default_1: "f32[8, 512, 30000]" = torch.ops.aten.view.default(addmm_74, _shape_param_0);  addmm_74 = _shape_param_0 = None
        view_default_2: "f32[4096, 30000]" = torch.ops.aten.view.default(view_default_1, _shape_param_1);  view_default_1 = _shape_param_1 = None
        amax_default: "f32[4096, 1]" = torch.ops.aten.amax.default(view_default_2, [1], True)
        sub_tensor: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(view_default_2, amax_default);  view_default_2 = amax_default = None
        exp_default: "f32[4096, 30000]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[4096, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar_1: "b8[4096]" = torch.ops.aten.ne.Scalar(view_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[4096]" = torch.ops.aten.where.self(ne_scalar_1, view_default, full_default);  ne_scalar_1 = full_default = None
        unsqueeze_default: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[4096]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[4096]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[4096]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  ne_scalar = neg_default = full_default_1 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        ne_scalar_2: "b8[4096]" = torch.ops.aten.ne.Scalar(view_default, -100);  view_default = None
        sum_default_1: "i64[]" = torch.ops.aten.sum.default(ne_scalar_2);  ne_scalar_2 = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_1, torch.float32);  sum_default_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default, convert_element_type_default);  sum_default = convert_element_type_default = None
        return div_tensor

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
