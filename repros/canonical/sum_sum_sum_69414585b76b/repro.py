"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train_001
Pattern hash: 69414585b76b
Shape hash: f9acc585
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
_shapes_config = "(T([16384, 768], f32), T([768], f32), T([128, 128, 768], f32), T([128, 128, 1], f32), T([128, 128, 768], f32), T([], f32), T([128, 128, 768], b8), S([128, 128, 768]), S([768]), S([768]), S([128, 128, 768]), S([16384, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_138: "f32[16384, 768]", arg7_1: "f32[768]", arg112_1: "f32[128, 128, 768]", arg111_1: "f32[128, 128, 1]", mul_355: "f32[128, 128, 768]", full_1: "f32[]", arg110_1: "b8[128, 128, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f32[128, 128, 768]" = torch.ops.aten.view.default(mm_138, _shape_param_0);  mm_138 = _shape_param_0 = None
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1], True)
        view_default_1: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        mul_tensor: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(arg7_1, arg112_1)
        add_tensor: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(arg111_1, 1e-06)
        div_tensor: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor, add_tensor);  mul_tensor = None
        div_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_tensor, add_tensor);  div_tensor = None
        neg_default: "f32[128, 128, 768]" = torch.ops.aten.neg.default(view_default)
        mul_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(view_default, add_tensor);  view_default = add_tensor = None
        sum_dim_int_list_1: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        mul_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_tensor_2, arg7_1);  arg7_1 = None
        mul_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_tensor_2, arg112_1);  div_tensor_2 = None
        sum_dim_int_list_2: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        view_default_2: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_2, _shape_param_2);  sum_dim_int_list_2 = _shape_param_2 = None
        neg_default_1: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_tensor_2)
        sum_dim_int_list_3: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_default_1, [2], True);  neg_default_1 = None
        add_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_355, mul_tensor_2);  mul_355 = mul_tensor_2 = None
        mul_scalar: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(arg111_1, 2)
        div_tensor_3: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_dim_int_list_1, mul_scalar);  sum_dim_int_list_1 = mul_scalar = None
        eq_scalar: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(arg111_1, 0);  arg111_1 = None
        where_self: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_scalar, full_1, div_tensor_3);  eq_scalar = full_1 = div_tensor_3 = None
        mul_scalar_1: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_self, 0.002607561929595828);  where_self = None
        mul_tensor_4: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_scalar_1, arg112_1);  mul_scalar_1 = arg112_1 = None
        add_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_4);  add_tensor_1 = mul_tensor_4 = None
        expand_default: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_dim_int_list_3, _shape_param_3);  sum_dim_int_list_3 = _shape_param_3 = None
        div_scalar: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        add_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, div_scalar);  add_tensor_2 = div_scalar = None
        convert_element_type_default: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(arg110_1, torch.float32);  arg110_1 = None
        mul_tensor_5: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_3, mul_tensor_5);  add_tensor_3 = mul_tensor_5 = None
        view_default_3: "f32[16384, 768]" = torch.ops.aten.view.default(mul_tensor_6, _shape_param_4);  mul_tensor_6 = _shape_param_4 = None
        permute_default: "f32[768, 16384]" = torch.ops.aten.permute.default(view_default_3, [1, 0])
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True);  view_default_3 = None
        view_default_4: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_5);  sum_dim_int_list_4 = _shape_param_5 = None
        return (view_default_1, view_default_2, permute_default, view_default_4)



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
