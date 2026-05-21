"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train_001
Pattern hash: b3091402bb28
Shape hash: b6788913
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([768], f32), T([128, 128, 768], f32), T([128, 128, 1], f32), T([128, 128, 768], f32), T([], f32), T([128, 128, 768], b8), T([128, 128], i64, gen=Index(3)), T([128, 128], i64, gen=Index(20005)), S([128, 128, 768]), S([128, 128, 768]), S([128, 128, 768]), S([768]), S([768]), S([128, 128, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_142: "f32[16384, 768]", mm_144: "f32[16384, 768]", mm_146: "f32[16384, 768]", arg2_1: "f32[768]", arg103_1: "f32[128, 128, 768]", arg102_1: "f32[128, 128, 1]", add_138: "f32[128, 128, 768]", full_1: "f32[]", arg101_1: "b8[128, 128, 768]", arg1_1: "i64[128, 128]", arg0_1: "i64[128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f32[128, 128, 768]" = torch.ops.aten.view.default(mm_142, _shape_param_0);  mm_142 = _shape_param_0 = None
        view_default_1: "f32[128, 128, 768]" = torch.ops.aten.view.default(mm_144, _shape_param_1);  mm_144 = _shape_param_1 = None
        add_tensor: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[128, 128, 768]" = torch.ops.aten.view.default(mm_146, _shape_param_2);  mm_146 = _shape_param_2 = None
        add_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1], True)
        view_default_3: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None
        mul_tensor: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(arg2_1, arg103_1)
        add_tensor_2: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(arg102_1, 1e-06)
        div_tensor: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor, add_tensor_2);  mul_tensor = None
        div_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_tensor, add_tensor_2);  div_tensor = None
        neg_default: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_tensor_1)
        mul_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None
        sum_dim_int_list_1: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        mul_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_tensor_2, arg2_1);  arg2_1 = None
        mul_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_tensor_2, arg103_1);  div_tensor_2 = None
        sum_dim_int_list_2: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        view_default_4: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_2, _shape_param_4);  sum_dim_int_list_2 = _shape_param_4 = None
        neg_default_1: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_tensor_2)
        sum_dim_int_list_3: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_default_1, [2], True);  neg_default_1 = None
        add_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_138, mul_tensor_2);  add_138 = mul_tensor_2 = None
        mul_scalar: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(arg102_1, 2)
        div_tensor_3: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_dim_int_list_1, mul_scalar);  sum_dim_int_list_1 = mul_scalar = None
        eq_scalar: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(arg102_1, 0);  arg102_1 = None
        where_self: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_scalar, full_1, div_tensor_3);  eq_scalar = div_tensor_3 = None
        mul_scalar_1: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_self, 0.002607561929595828);  where_self = None
        mul_tensor_4: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_scalar_1, arg103_1);  mul_scalar_1 = arg103_1 = None
        add_tensor_4: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_4);  add_tensor_3 = mul_tensor_4 = None
        expand_default: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_dim_int_list_3, _shape_param_5);  sum_dim_int_list_3 = _shape_param_5 = None
        div_scalar: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        add_tensor_5: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_4, div_scalar);  add_tensor_4 = div_scalar = None
        convert_element_type_default: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(arg101_1, torch.float32);  arg101_1 = None
        mul_tensor_5: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_5, mul_tensor_5);  add_tensor_5 = mul_tensor_5 = None
        eq_scalar_1: "b8[128, 128]" = torch.ops.aten.eq.Scalar(arg1_1, 0)
        unsqueeze_default: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[128, 128, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_1, mul_tensor_6);  unsqueeze_default = None
        full_default: "f32[3, 768]" = torch.ops.aten.full.default([3, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[3, 768]" = torch.ops.aten.index_put.default(full_default, [arg1_1], where_self_1, True);  full_default = arg1_1 = where_self_1 = None
        eq_scalar_2: "b8[128, 128]" = torch.ops.aten.eq.Scalar(arg0_1, 0)
        unsqueeze_default_1: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[128, 128, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_1, mul_tensor_6);  unsqueeze_default_1 = full_1 = mul_tensor_6 = None
        full_default_1: "f32[20005, 768]" = torch.ops.aten.full.default([20005, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[20005, 768]" = torch.ops.aten.index_put.default(full_default_1, [arg0_1], where_self_2, True);  full_default_1 = arg0_1 = where_self_2 = None
        return (view_default_3, view_default_4, index_put_default, index_put_default_1)



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
