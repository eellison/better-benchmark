"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_train_001
Pattern hash: 6d6ec1e4f644
Shape hash: d77624b3
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
_shapes_config = "(T([8192, 768], f32), T([768], f32), T([8, 1024, 768], f32), T([8, 1024, 1], f32), T([8, 1024, 768], b8), S([8, 1024, 768]), S([8192, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_1: "f32[8192, 768]", arg73_1: "f32[768]", arg226_1: "f32[8, 1024, 768]", arg233_1: "f32[8, 1024, 1]", arg225_1: "b8[8, 1024, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_0);  mm_1 = _shape_param_0 = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_default, arg73_1);  arg73_1 = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg226_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg226_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg233_1, sub_tensor_1);  arg233_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_default, arg226_1);  arg226_1 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        convert_element_type_default: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg225_1, torch.float32);  arg225_1 = None
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None
        view_default_1: "f32[8192, 768]" = torch.ops.aten.view.default(mul_tensor_7, _shape_param_1);  mul_tensor_7 = _shape_param_1 = None
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True);  view_default_1 = None
        view_default_2: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, view_default_2)



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
