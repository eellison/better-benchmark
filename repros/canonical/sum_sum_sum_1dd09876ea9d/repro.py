"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train_001
Pattern hash: 1dd09876ea9d
Shape hash: 03c36796
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
_shapes_config = "(T([128, 512, 12, 12], f32), T([128, 512, 24, 24], f32), T([128, 512, 24, 24], f32), T([128, 512, 24, 24], f32), T([128, 512, 1, 1], f32), T([128, 512, 24, 24], f32), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_168: "f32[128, 512, 12, 12]", arg231_1: "f32[128, 512, 24, 24]", getitem_165: "f32[128, 512, 24, 24]", arg460_1: "f32[128, 512, 24, 24]", arg230_1: "f32[128, 512, 1, 1]", arg227_1: "f32[128, 512, 24, 24]", arg44_1: "f32[]"):
        # No stacktrace found for following nodes
        avg_pool2d_backward_default: "f32[128, 512, 24, 24]" = torch.ops.aten.avg_pool2d_backward.default(getitem_168, arg231_1, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_168 = arg231_1 = None
        add_tensor: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(getitem_165, avg_pool2d_backward_default);  getitem_165 = avg_pool2d_backward_default = None
        mul_tensor: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, 0.9622504486493761);  add_tensor = None
        mul_tensor_1: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.7015043497085571);  mul_tensor = None
        mul_tensor_2: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg460_1);  mul_tensor_1 = arg460_1 = None
        mul_tensor_3: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.2);  mul_tensor_2 = None
        sigmoid_default: "f32[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(arg230_1);  arg230_1 = None
        mul_tensor_4: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(arg227_1, sigmoid_default)
        mul_tensor_5: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 2.0);  mul_tensor_4 = None
        mul_tensor_6: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_7: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_3, arg44_1);  mul_tensor_3 = arg44_1 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(mul_tensor_6);  mul_tensor_6 = None
        mul_tensor_8: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 2.0);  mul_tensor_7 = None
        mul_tensor_9: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_8, arg227_1);  mul_tensor_8 = arg227_1 = None
        sum_dim_int_list: "f32[128, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [2, 3], True);  mul_tensor_9 = None
        sub_tensor: "f32[128, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_10: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor);  sigmoid_default = sub_tensor = None
        mul_tensor_11: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_10);  sum_dim_int_list = mul_tensor_10 = None
        sum_dim_int_list_1: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 2, 3]);  mul_tensor_11 = None
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
