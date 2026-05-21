"""
Standalone repro captured via capture_hook.
Label: timm_timm_vit_base_patch16_siglip_256_train_train_001
Pattern hash: 26d3d5760c30
Shape hash: 0a8651c2
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
_shapes_config = "(T([32768, 768], f32), T([768], f32), T([128, 768, 16, 16], f32), T([1, 256, 768], f32), T([128, 256, 1], f32), T([128, 256, 1], f32), T([128, 256, 768], f32), S([128, 256, 768]), S([128, 768, 256]), S([128, 768, 16, 16]))"

class Repro(torch.nn.Module):
    def forward(self, mm_104: "f32[32768, 768]", arg3_1: "f32[768]", arg83_1: "f32[128, 768, 16, 16]", arg2_1: "f32[1, 256, 768]", arg84_1: "f32[128, 256, 1]", arg85_1: "f32[128, 256, 1]", add_49: "f32[128, 256, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[128, 256, 768]" = torch.ops.aten.view.default(mm_104, _shape_param_0);  mm_104 = _shape_param_0 = None
        mul_tensor: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(view_default, arg3_1);  view_default = arg3_1 = None
        mul_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        view_default_1: "f32[128, 768, 256]" = torch.ops.aten.view.default(arg83_1, _shape_param_1);  arg83_1 = _shape_param_1 = None
        permute_default: "f32[128, 256, 768]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1]);  view_default_1 = None
        add_tensor: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(permute_default, arg2_1);  permute_default = arg2_1 = None
        sub_tensor: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(add_tensor, arg84_1);  add_tensor = arg84_1 = None
        mul_tensor_2: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, arg85_1);  sub_tensor = None
        mul_tensor_3: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(arg85_1, 768);  arg85_1 = None
        mul_tensor_5: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(add_49, mul_tensor_5);  add_49 = mul_tensor_5 = None
        permute_default_1: "f32[128, 768, 256]" = torch.ops.aten.permute.default(add_tensor_1, [0, 2, 1]);  add_tensor_1 = None
        view_default_2: "f32[128, 768, 16, 16]" = torch.ops.aten.view.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None
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
