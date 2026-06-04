"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train_001
Pattern hash: 3ec568a7ba04
Shape hash: 102f3a82
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 1536, 6, 6], f32), T([128, 1536, 1, 1], f32), T([128, 1536, 6, 6], f32), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[128, 1536, 6, 6]", arg409_1: "f32[128, 1536, 1, 1]", arg406_1: "f32[128, 1536, 6, 6]", arg147_1: "f32[]"):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(getitem, 0.2);  getitem = None
        sigmoid_default: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(arg409_1);  arg409_1 = None
        mul_tensor_1: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(arg406_1, sigmoid_default)
        mul_tensor_2: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 2.0);  mul_tensor_1 = None
        mul_tensor_3: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_4: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor, arg147_1);  mul_tensor = arg147_1 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_5: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 2.0);  mul_tensor_4 = None
        mul_tensor_6: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_5, arg406_1);  mul_tensor_5 = arg406_1 = None
        sum_dim_int_list: "f32[128, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2, 3], True);  mul_tensor_6 = None
        sub_tensor: "f32[128, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_7: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor);  sigmoid_default = sub_tensor = None
        mul_tensor_8: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_7);  sum_dim_int_list = mul_tensor_7 = None
        sum_dim_int_list_1: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        return (sum_default, sum_dim_int_list_1)

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
