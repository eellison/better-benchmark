"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_regnet_train_001
Pattern hash: 4dbee302662f
Shape hash: da868e19
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
_shapes_config = "(T([32, 224, 56, 56], f32), T([1, 224, 1, 1], f32), T([1, 224, 1, 1], f32), T([224], f32), T([224], f32), T([32, 224, 56, 56], f32), T([32, 224, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg191_1: "f32[32, 224, 56, 56]", arg192_1: "f32[1, 224, 1, 1]", arg193_1: "f32[1, 224, 1, 1]", arg6_1: "f32[224]", arg7_1: "f32[224]", getitem_282: "f32[32, 224, 56, 56]", arg196_1: "f32[32, 224, 1, 1]"):
        # No stacktrace found for following nodes
        sub_tensor: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(arg191_1, arg192_1);  arg191_1 = arg192_1 = None
        mul_tensor: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, arg193_1);  sub_tensor = arg193_1 = None
        unsqueeze_default: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_default_1: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_default_3: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[32, 224, 56, 56]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        mul_tensor_2: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_282, relu_default);  getitem_282 = relu_default = None
        sigmoid_default: "f32[32, 224, 1, 1]" = torch.ops.aten.sigmoid.default(arg196_1);  arg196_1 = None
        sum_dim_int_list: "f32[32, 224, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True);  mul_tensor_2 = None
        sub_tensor_1: "f32[32, 224, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_3: "f32[32, 224, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1);  sigmoid_default = sub_tensor_1 = None
        mul_tensor_4: "f32[32, 224, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_3);  sum_dim_int_list = mul_tensor_3 = None
        sum_dim_int_list_1: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3]);  mul_tensor_4 = None
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
