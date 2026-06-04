"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train_001
Pattern hash: ece81a19042b
Shape hash: 3b285a26
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 16, 112, 112], f32), T([512, 8, 112, 112], f32), T([512, 8, 112, 112], f32), T([1, 8, 1, 1], f32), T([8], f32), T([8], f32))"

class Repro(torch.nn.Module):
    def forward(self, copy_25: "f32[512, 16, 112, 112]", getitem_270: "f32[512, 8, 112, 112]", arg206_1: "f32[512, 8, 112, 112]", arg537_1: "f32[1, 8, 1, 1]", arg207_1: "f32[8]", arg9_1: "f32[8]"):
        # No stacktrace found for following nodes
        slice_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.slice.Tensor(copy_25, 1, 0, 8);  copy_25 = None
        add_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.add.Tensor(slice_tensor, getitem_270);  slice_tensor = getitem_270 = None
        sum_dim_int_list: "f32[8]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 2, 3])
        sub_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.sub.Tensor(arg206_1, arg537_1);  arg206_1 = arg537_1 = None
        mul_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor)
        sum_dim_int_list_1: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[8]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 1.5570192920918366e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[8]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 1.5570192920918366e-07)
        mul_tensor_3: "f32[8]" = torch.ops.aten.mul.Tensor(arg207_1, arg207_1)
        mul_tensor_4: "f32[8]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[8]" = torch.ops.aten.mul.Tensor(arg207_1, arg9_1);  arg9_1 = None
        unsqueeze_default_6: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 8, 112, 112]" = torch.ops.aten.sub.Tensor(add_tensor, mul_tensor_6);  add_tensor = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 8, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[8]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg207_1);  sum_dim_int_list_1 = arg207_1 = None
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
