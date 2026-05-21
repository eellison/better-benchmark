"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train_001
Pattern hash: dd926ede2e30
Shape hash: a4886f9d
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
_shapes_config = "(T([128, 144, 32, 32], f32), T([144], f32), T([512, 256, 144], f32), T([512, 256, 1], f32), S([294912, 2, 16, 2]), S([128, 144, 256, 4]), S([512, 256, 144]), S([131072, 144]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_76: "f32[128, 144, 32, 32]", arg60_1: "f32[144]", arg240_1: "f32[512, 256, 144]", arg413_1: "f32[512, 256, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        clone_default: "f32[128, 144, 32, 32]" = torch.ops.aten.clone.default(getitem_76, memory_format = torch.contiguous_format);  getitem_76 = None
        view_default: "f32[294912, 2, 16, 2]" = torch.ops.aten.view.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        permute_default: "f32[294912, 16, 2, 2]" = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3]);  view_default = None
        clone_default_1: "f32[294912, 16, 2, 2]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1: "f32[128, 144, 256, 4]" = torch.ops.aten.view.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        permute_default_1: "f32[128, 4, 256, 144]" = torch.ops.aten.permute.default(view_default_1, [0, 3, 2, 1]);  view_default_1 = None
        clone_default_2: "f32[128, 4, 256, 144]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_2: "f32[512, 256, 144]" = torch.ops.aten.view.default(clone_default_2, _shape_param_2);  clone_default_2 = _shape_param_2 = None
        mul_tensor: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(view_default_2, arg60_1);  view_default_2 = arg60_1 = None
        mul_tensor_1: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_tensor, 144)
        sum_dim_int_list: "f32[512, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_tensor, arg240_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[512, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(arg240_1, sum_dim_int_list_1);  arg240_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(arg413_1, sub_tensor_1);  arg413_1 = sub_tensor_1 = None
        view_default_3: "f32[131072, 144]" = torch.ops.aten.view.default(mul_tensor_4, _shape_param_3);  mul_tensor_4 = _shape_param_3 = None
        return view_default_3



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
