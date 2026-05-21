"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_train_007
Pattern hash: 2774290d7890
Shape hash: da0c63d9
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
_shapes_config = "(T([4096, 2560], f32), T([2560], f32), T([4096, 2560], f32), T([32, 128, 2560], b8), T([32, 128, 2560], f32), T([32, 128, 1], f32), T([32, 128, 1], f32), T([32, 128, 2560], f32), S([32, 128, 2560]), S([32, 128, 2560]), S([4096, 2560]))"

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[4096, 2560]", arg6_1: "f32[2560]", arg20_1: "f32[4096, 2560]", arg21_1: "b8[32, 128, 2560]", arg1_1: "f32[32, 128, 2560]", arg22_1: "f32[32, 128, 1]", arg23_1: "f32[32, 128, 1]", arg28_1: "f32[32, 128, 2560]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[32, 128, 2560]" = torch.ops.aten.view.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None
        mul_tensor: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(view_default, arg6_1);  view_default = arg6_1 = None
        mul_tensor_1: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, 2560)
        sum_dim_int_list: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        view_default_1: "f32[32, 128, 2560]" = torch.ops.aten.view.default(arg20_1, _shape_param_1);  arg20_1 = _shape_param_1 = None
        mul_tensor_2: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(arg21_1, view_default_1);  view_default_1 = None
        mul_tensor_3: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        add_tensor: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(arg1_1, mul_tensor_3);  arg1_1 = mul_tensor_3 = None
        sub_tensor: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(add_tensor, arg22_1);  add_tensor = arg22_1 = None
        mul_tensor_4: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, arg23_1);  sub_tensor = None
        mul_tensor_5: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  mul_tensor_4 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_6);  sub_tensor_1 = mul_tensor_6 = None
        div_tensor: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(arg23_1, 2560);  arg23_1 = None
        mul_tensor_7: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor_1: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(arg28_1, mul_tensor_7);  arg28_1 = mul_tensor_7 = None
        convert_element_type_default: "f32[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.float32);  arg21_1 = None
        mul_tensor_8: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_9: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_8);  add_tensor_1 = mul_tensor_8 = None
        view_default_2: "f32[4096, 2560]" = torch.ops.aten.view.default(mul_tensor_9, _shape_param_2);  mul_tensor_9 = _shape_param_2 = None
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
