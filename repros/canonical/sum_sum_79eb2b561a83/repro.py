"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train_001
Pattern hash: 79eb2b561a83
Shape hash: feb55e15
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
_shapes_config = "(T([512, 48, 56, 56], f32), T([512, 48, 56, 56], f32), T([1, 48, 1, 1], f32), T([48], f32), T([48], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_258: "f32[512, 48, 56, 56]", arg219_1: "f32[512, 48, 56, 56]", arg534_1: "f32[1, 48, 1, 1]", arg220_1: "f32[48]", arg18_1: "f32[48]"):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[48]" = torch.ops.aten.sum.dim_IntList(getitem_258, [0, 2, 3])
        sub_tensor: "f32[512, 48, 56, 56]" = torch.ops.aten.sub.Tensor(arg219_1, arg534_1);  arg219_1 = arg534_1 = None
        mul_tensor: "f32[512, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_258, sub_tensor)
        sum_dim_int_list_1: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[48]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.228077168367346e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[48]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.228077168367346e-07)
        mul_tensor_3: "f32[48]" = torch.ops.aten.mul.Tensor(arg220_1, arg220_1)
        mul_tensor_4: "f32[48]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[48]" = torch.ops.aten.mul.Tensor(arg220_1, arg18_1);  arg18_1 = None
        unsqueeze_default_6: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 48, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_258, mul_tensor_6);  getitem_258 = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[48]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg220_1);  sum_dim_int_list_1 = arg220_1 = None
        return (mul_tensor_7, mul_tensor_8)



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
