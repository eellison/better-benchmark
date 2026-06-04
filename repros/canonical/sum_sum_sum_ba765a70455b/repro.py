"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_001
Pattern hash: ba765a70455b
Shape hash: 34e07cb7
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([25088, 512], f32), T([512], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 1], f32), T([128, 14, 14, 512], f32), T([128, 1, 1], b8), S([512, 49, 512]), S([512, 7, 7, 512]), S([128, 2, 2, 7, 7, 512]), S([128, 14, 14, 512]), S([128, 196, 512]), S([25088, 512]), S([512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_146: "f32[25088, 512]", arg50_1: "f32[512]", arg253_1: "f32[128, 14, 14, 512]", arg529_1: "f32[128, 14, 14, 1]", view_575: "f32[128, 14, 14, 512]", arg252_1: "b8[128, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view_default: "f32[512, 49, 512]" = torch.ops.aten.view.default(mm_146, _shape_param_0);  mm_146 = _shape_param_0 = None
        view_default_1: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        view_default_2: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 2, 4, 5]);  view_default_2 = None
        clone_default: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        mul_tensor: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_default_3, arg50_1);  arg50_1 = None
        mul_tensor_1: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 512)
        sum_dim_int_list: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg253_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(arg253_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(arg529_1, sub_tensor_1);  arg529_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_default_3, arg253_1);  arg253_1 = None
        sum_dim_int_list_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1, 2]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_default_3, [0, 1, 2]);  view_default_3 = None
        add_tensor: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_575, mul_tensor_4);  view_575 = mul_tensor_4 = None
        view_default_4: "f32[128, 196, 512]" = torch.ops.aten.view.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None
        convert_element_type_default: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.float32);  arg252_1 = None
        div_tensor: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9782608672976494);  convert_element_type_default = None
        mul_tensor_6: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_default_4, div_tensor);  view_default_4 = div_tensor = None
        view_default_5: "f32[25088, 512]" = torch.ops.aten.view.default(mul_tensor_6, _shape_param_5);  mul_tensor_6 = _shape_param_5 = None
        permute_default_1: "f32[512, 25088]" = torch.ops.aten.permute.default(view_default_5, [1, 0])
        sum_dim_int_list_4: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_default_5, [0], True);  view_default_5 = None
        view_default_6: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_6);  sum_dim_int_list_4 = _shape_param_6 = None
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
