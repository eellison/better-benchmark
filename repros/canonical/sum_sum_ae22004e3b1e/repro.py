"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train_001
Pattern hash: ae22004e3b1e
Shape hash: 72af9b68
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 1536, 7, 7], f32), T([128, 1536, 7, 7], f32), T([128, 1536, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[128, 1536, 7, 7]", arg401_1: "f32[128, 1536, 7, 7]", arg404_1: "f32[128, 1536, 1, 1]"):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, 0.2);  getitem = None
        mul_tensor_1: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        mul_tensor_2: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg401_1);  mul_tensor_1 = arg401_1 = None
        sigmoid_default: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(arg404_1);  arg404_1 = None
        sum_dim_int_list: "f32[128, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True);  mul_tensor_2 = None
        sub_tensor: "f32[128, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_3: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor);  sigmoid_default = sub_tensor = None
        mul_tensor_4: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_3);  sum_dim_int_list = mul_tensor_3 = None
        sum_dim_int_list_1: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3]);  mul_tensor_4 = None
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
