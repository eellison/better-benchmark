"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_DistilBert_train_001
Pattern hash: 502c40da68e8
Shape hash: ab68fd7c
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([96, 512, 512], f32), T([8, 12, 512, 512], b8), T([1, 1, 512, 1], b8), T([], f32), T([96, 512, 512], f32), T([8, 12, 512, 1], f32), T([8, 12, 512, 1], f32), T([8, 12, 512, 1], b8), S([8, 12, 512, 512]), S([8, -1, 512, 512]), S([8, 12, 512, 512]), S([96, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_21: "f32[96, 512, 512]", arg66_1: "b8[8, 12, 512, 512]", arg60_1: "b8[1, 1, 512, 1]", full_1: "f32[]", arg62_1: "f32[96, 512, 512]", arg63_1: "f32[8, 12, 512, 1]", arg64_1: "f32[8, 12, 512, 1]", arg65_1: "b8[8, 12, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[8, 12, 512, 512]" = torch.ops.aten.view.default(bmm_21, _shape_param_0);  bmm_21 = _shape_param_0 = None
        convert_element_type_default: "f32[8, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg66_1, torch.float32);  arg66_1 = None
        mul_tensor: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  view_default = mul_tensor = None
        expand_default: "b8[8, 1, 512, 512]" = torch.ops.aten.expand.default(arg60_1, _shape_param_1);  arg60_1 = _shape_param_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_1, full_default);  expand_default = full_1 = full_default = None
        view_default_1: "f32[8, 12, 512, 512]" = torch.ops.aten.view.default(arg62_1, _shape_param_2);  arg62_1 = _shape_param_2 = None
        add_tensor: "f32[8, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_default_1, where_self);  view_default_1 = where_self = None
        sub_tensor: "f32[8, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_tensor, arg63_1);  add_tensor = arg63_1 = None
        exp_default: "f32[8, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor: "f32[8, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, arg64_1);  exp_default = arg64_1 = None
        full_default_1: "f32[8, 12, 512, 512]" = torch.ops.aten.full.default([8, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8, 12, 512, 512]" = torch.ops.aten.where.self(arg65_1, full_default_1, div_tensor);  arg65_1 = full_default_1 = div_tensor = None
        mul_tensor_2: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, where_self_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[8, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[8, 12, 512, 512]" = torch.ops.aten.neg.default(where_self_1);  where_self_1 = None
        fma_default: "f32[8, 12, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        view_default_2: "f32[96, 512, 512]" = torch.ops.aten.view.default(fma_default, _shape_param_3);  fma_default = _shape_param_3 = None
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
