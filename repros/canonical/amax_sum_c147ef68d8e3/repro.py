"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer_005
Pattern hash: c147ef68d8e3
Shape hash: 26280f06
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device
torch._inductor.config.force_pointwise_cat = True

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1, 12, 4096], i64), T([768, 64, 128], f16), T([], f16), S([1, 12, -1, 64]), S([1, 12, 64, 64, 128]), S([1, 12, 64, 64, 128]), S([768, 64, 128]))"

class Repro(torch.nn.Module):
    def forward(self, remainder_5: "i64[1, 12, 4096]", bmm_13: "f16[768, 64, 128]", arg67_1: "f16[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "i64[1, 12, 64, 64]" = torch.ops.aten.view.default(remainder_5, _shape_param_0);  remainder_5 = _shape_param_0 = None
        unsqueeze_default: "i64[1, 12, 64, 64, 1]" = torch.ops.aten.unsqueeze.default(view_default, -1)
        slice_tensor: "i64[1, 12, 1, 64]" = torch.ops.aten.slice.Tensor(view_default, 2, -1, 9223372036854775807)
        slice_tensor_1: "i64[1, 12, 63, 64]" = torch.ops.aten.slice.Tensor(view_default, 2, 0, -1)
        cat_default: "i64[1, 12, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None
        cat_default_1: "i64[1, 12, 64, 128]" = torch.ops.aten.cat.default([cat_default, view_default], 3);  cat_default = view_default = None
        unsqueeze_default_1: "i64[1, 12, 64, 1, 128]" = torch.ops.aten.unsqueeze.default(cat_default_1, -2);  cat_default_1 = None
        ne_tensor: "b8[1, 12, 64, 64, 128]" = torch.ops.aten.ne.Tensor(unsqueeze_default, unsqueeze_default_1);  unsqueeze_default = unsqueeze_default_1 = None
        view_default_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.view.default(bmm_13, _shape_param_1);  bmm_13 = _shape_param_1 = None
        where_self: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.where.self(ne_tensor, view_default_1, arg67_1);  ne_tensor = view_default_1 = arg67_1 = None
        convert_element_type_default: "f32[1, 12, 64, 64, 128]" = torch.ops.prims.convert_element_type.default(where_self, torch.float32)
        amax_default: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        abs_default: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax_default)
        eq_scalar: "b8[1, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_default, inf);  abs_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq_scalar, full_default, amax_default);  eq_scalar = full_default = amax_default = None
        sub_tensor: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default, where_self_1);  convert_element_type_default = None
        exp_default: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True);  exp_default = None
        log_default: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        add_tensor: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log_default, where_self_1);  log_default = where_self_1 = None
        convert_element_type_default_1: "f16[1, 12, 64, 64, 1]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float16);  add_tensor = None
        sub_tensor_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(where_self, convert_element_type_default_1);  where_self = convert_element_type_default_1 = None
        exp_default_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        expand_default: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_default_1, _shape_param_2);  exp_default_1 = _shape_param_2 = None
        view_default_2: "f16[768, 64, 128]" = torch.ops.aten.view.default(expand_default, _shape_param_3);  expand_default = _shape_param_3 = None
        return view_default_2

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
