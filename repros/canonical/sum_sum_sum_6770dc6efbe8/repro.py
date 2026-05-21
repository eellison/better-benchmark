"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_001
Pattern hash: 6770dc6efbe8
Shape hash: f7678e55
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
_shapes_config = "(T([100352, 512], f32), T([512], f32), T([128, 28, 28, 512], f32), T([128, 28, 28, 1], f32), T([128, 1, 1], b8), S([128, 28, 28, 512]), S([128, 28, 28, 2, 2, 128]), S([128, 56, 56, 128]), S([128, 3136, 128]), S([401408, 128]), S([128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_183: "f32[100352, 512]", arg18_1: "f32[512]", arg203_1: "f32[128, 28, 28, 512]", arg549_1: "f32[128, 28, 28, 1]", arg202_1: "b8[128, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f32[128, 28, 28, 512]" = torch.ops.aten.view.default(mm_183, _shape_param_0);  mm_183 = _shape_param_0 = None
        mul_tensor: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(view_default, arg18_1);  arg18_1 = None
        mul_tensor_1: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 512)
        sum_dim_int_list: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg203_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(arg203_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(arg549_1, sub_tensor_1);  arg549_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(view_default, arg203_1);  arg203_1 = None
        sum_dim_int_list_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1, 2]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1, 2]);  view_default = None
        view_default_1: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.view.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None
        permute_default: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.permute.default(view_default_1, [0, 1, 4, 2, 3, 5]);  view_default_1 = None
        clone_default: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_2: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        view_default_3: "f32[128, 3136, 128]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        convert_element_type_default: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg202_1, torch.float32);  arg202_1 = None
        div_tensor: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9956521736457944);  convert_element_type_default = None
        mul_tensor_6: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view_default_3, div_tensor);  view_default_3 = div_tensor = None
        view_default_4: "f32[401408, 128]" = torch.ops.aten.view.default(mul_tensor_6, _shape_param_4);  mul_tensor_6 = _shape_param_4 = None
        permute_default_1: "f32[128, 401408]" = torch.ops.aten.permute.default(view_default_4, [1, 0])
        sum_dim_int_list_4: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_default_4, [0], True);  view_default_4 = None
        view_default_5: "f32[128]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_5);  sum_dim_int_list_4 = _shape_param_5 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default_1, view_default_5)



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
