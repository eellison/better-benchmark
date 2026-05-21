"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train_001
Pattern hash: 66d579ce122c
Shape hash: ea4a2f54
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
_shapes_config = "(T([128, 32, 112, 112], f32), T([1, 32, 1, 1], f32), T([1, 32, 1, 1], f32), T([32], f32), T([32], f32), T([128, 32, 112, 112], f32), T([128, 32, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg169_1: "f32[128, 32, 112, 112]", arg170_1: "f32[1, 32, 1, 1]", arg171_1: "f32[1, 32, 1, 1]", arg4_1: "f32[32]", arg5_1: "f32[32]", getitem_228: "f32[128, 32, 112, 112]", arg175_1: "f32[128, 32, 1, 1]"):
        # No stacktrace found for following nodes
        sub_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(arg169_1, arg170_1);  arg169_1 = arg170_1 = None
        mul_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, arg171_1);  sub_tensor = arg171_1 = None
        unsqueeze_default: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_default_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_default_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        neg_default: "f32[128, 32, 112, 112]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[128, 32, 112, 112]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.div.Tensor(add_tensor, add_tensor_1);  add_tensor = add_tensor_1 = None
        mul_tensor_2: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_228, div_tensor);  getitem_228 = div_tensor = None
        sigmoid_default: "f32[128, 32, 1, 1]" = torch.ops.aten.sigmoid.default(arg175_1);  arg175_1 = None
        sum_dim_int_list: "f32[128, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True);  mul_tensor_2 = None
        sub_tensor_1: "f32[128, 32, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_3: "f32[128, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1);  sigmoid_default = sub_tensor_1 = None
        mul_tensor_4: "f32[128, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_3);  sum_dim_int_list = mul_tensor_3 = None
        sum_dim_int_list_1: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3]);  mul_tensor_4 = None
        return sum_dim_int_list_1



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
