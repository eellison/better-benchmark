"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_001
Pattern hash: 1cad42039094
Shape hash: 8b656d5d
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
_shapes_config = "(T([100352, 256], f32), T([256], f32), T([128, 784, 256], f32), T([128, 784, 1], f32), T([128, 784, 256], f32), T([128, 1, 1, 1], b8), S([128, 784, 256]), S([128, 28, 28, 256]), S([128, 4, 7, 4, 7, 256]), S([2048, 7, 7, 256]), S([2048, 49, 256]), S([100352, 256]), S([256]))"

class Repro(torch.nn.Module):
    def forward(self, mm_176: "f32[100352, 256]", arg24_1: "f32[256]", arg212_1: "f32[128, 784, 256]", arg545_1: "f32[128, 784, 1]", view_704: "f32[128, 784, 256]", arg211_1: "b8[128, 1, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view_default: "f32[128, 784, 256]" = torch.ops.aten.view.default(mm_176, _shape_param_0);  mm_176 = _shape_param_0 = None
        mul_tensor: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_default, arg24_1);  arg24_1 = None
        mul_tensor_1: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, 256)
        sum_dim_int_list: "f32[128, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, arg212_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(arg212_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(arg545_1, sub_tensor_1);  arg545_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_default, arg212_1);  arg212_1 = None
        sum_dim_int_list_2: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        add_tensor: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(view_704, mul_tensor_4);  view_704 = mul_tensor_4 = None
        view_default_1: "f32[128, 28, 28, 256]" = torch.ops.aten.view.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        convert_element_type_default: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg211_1, torch.float32);  arg211_1 = None
        div_tensor: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9913043472915888);  convert_element_type_default = None
        mul_tensor_6: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_default_1, div_tensor);  view_default_1 = div_tensor = None
        view_default_2: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(mul_tensor_6, _shape_param_2);  mul_tensor_6 = _shape_param_2 = None
        permute_default: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 2, 4, 5]);  view_default_2 = None
        clone_default: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3: "f32[2048, 7, 7, 256]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        view_default_4: "f32[2048, 49, 256]" = torch.ops.aten.view.default(view_default_3, _shape_param_4);  view_default_3 = _shape_param_4 = None
        view_default_5: "f32[100352, 256]" = torch.ops.aten.view.default(view_default_4, _shape_param_5);  view_default_4 = _shape_param_5 = None
        permute_default_1: "f32[256, 100352]" = torch.ops.aten.permute.default(view_default_5, [1, 0])
        sum_dim_int_list_4: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_default_5, [0], True);  view_default_5 = None
        view_default_6: "f32[256]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_6);  sum_dim_int_list_4 = _shape_param_6 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default_1, view_default_6)



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
