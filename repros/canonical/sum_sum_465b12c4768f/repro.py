"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_001
Pattern hash: 465b12c4768f
Shape hash: 7538f768
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
_shapes_config = "(T([6272, 1024], f32), T([1024], f32), T([128, 7, 7, 1024], f32), T([128, 7, 7, 1], f32), T([128, 7, 7, 1024], f32), T([128, 1, 1], b8), S([128, 49, 1024]), S([128, 7, 7, 1024]), S([128, 1, 1, 7, 7, 1024]), S([128, 7, 7, 1024]), S([128, 49, 1024]), S([6272, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_8: "f32[6272, 1024]", arg171_1: "f32[1024]", arg427_1: "f32[128, 7, 7, 1024]", arg444_1: "f32[128, 7, 7, 1]", view_9: "f32[128, 7, 7, 1024]", arg426_1: "b8[128, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f32[128, 49, 1024]" = torch.ops.aten.view.default(mm_8, _shape_param_0);  mm_8 = _shape_param_0 = None
        view_default_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        view_default_2: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 2, 4, 5]);  view_default_2 = None
        view_default_3: "f32[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_default, _shape_param_3);  permute_default = _shape_param_3 = None
        mul_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_default_3, arg171_1);  view_default_3 = arg171_1 = None
        mul_tensor_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg427_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(arg427_1, sum_dim_int_list_1);  arg427_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(arg444_1, sub_tensor_1);  arg444_1 = sub_tensor_1 = None
        add_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_9, mul_tensor_4);  view_9 = mul_tensor_4 = None
        view_default_4: "f32[128, 49, 1024]" = torch.ops.aten.view.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None
        convert_element_type_default: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg426_1, torch.float32);  arg426_1 = None
        div_tensor: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9043478220701218);  convert_element_type_default = None
        mul_tensor_5: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_default_4, div_tensor);  view_default_4 = div_tensor = None
        view_default_5: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_5);  mul_tensor_5 = _shape_param_5 = None
        return view_default_5



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
