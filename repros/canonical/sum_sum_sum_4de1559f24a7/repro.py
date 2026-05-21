"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_train_001
Pattern hash: 4de1559f24a7
Shape hash: 9c069f54
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
_shapes_config = "(T([8192, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), T([1024], f32), T([16, 512, 1024], f32), T([16, 512, 1], f32), T([16, 512, 1024], f32), T([16, 512, 1024], b8), S([16, 512, 1024]), S([16, 512, 1024]), S([16, 512, 1024]), S([8192, 1024]), S([1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_274: "f32[8192, 1024]", mm_276: "f32[8192, 1024]", mm_278: "f32[8192, 1024]", arg11_1: "f32[1024]", arg216_1: "f32[16, 512, 1024]", arg641_1: "f32[16, 512, 1]", add_137: "f32[16, 512, 1024]", arg215_1: "b8[16, 512, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[16, 512, 1024]" = torch.ops.aten.view.default(mm_274, _shape_param_0);  mm_274 = _shape_param_0 = None
        view_default_1: "f32[16, 512, 1024]" = torch.ops.aten.view.default(mm_276, _shape_param_1);  mm_276 = _shape_param_1 = None
        add_tensor: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[16, 512, 1024]" = torch.ops.aten.view.default(mm_278, _shape_param_2);  mm_278 = _shape_param_2 = None
        add_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        mul_tensor: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg11_1);  arg11_1 = None
        mul_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg216_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg216_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg641_1, sub_tensor_1);  arg641_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg216_1);  arg216_1 = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1]);  add_tensor_1 = None
        add_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_137, mul_tensor_4);  add_137 = mul_tensor_4 = None
        convert_element_type_default: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(arg215_1, torch.float32);  arg215_1 = None
        mul_tensor_6: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_7: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_6);  add_tensor_2 = mul_tensor_6 = None
        view_default_3: "f32[8192, 1024]" = torch.ops.aten.view.default(mul_tensor_7, _shape_param_3);  mul_tensor_7 = _shape_param_3 = None
        permute_default: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_default_3, [1, 0])
        sum_dim_int_list_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True);  view_default_3 = None
        view_default_4: "f32[1024]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_4);  sum_dim_int_list_4 = _shape_param_4 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, view_default_4)



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
