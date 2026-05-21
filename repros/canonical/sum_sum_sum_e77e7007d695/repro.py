"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train_001
Pattern hash: e77e7007d695
Shape hash: e1a59f11
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
_shapes_config = "(T([128, 512, 24, 24], f32), T([128, 512, 1, 1], f32), T([128, 512, 24, 24], f32), T([], f32), T([128, 512, 24, 24], f32), T([128, 512, 24, 24], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_186: "f32[128, 512, 24, 24]", arg211_1: "f32[128, 512, 1, 1]", arg208_1: "f32[128, 512, 24, 24]", arg33_1: "f32[]", arg193_1: "f32[128, 512, 24, 24]", mul_785: "f32[128, 512, 24, 24]"):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_186, 0.9805806756909201);  getitem_186 = None
        mul_tensor_1: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.7015043497085571);  mul_tensor = None
        sigmoid_default: "f32[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(arg211_1);  arg211_1 = None
        mul_tensor_2: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(arg208_1, sigmoid_default)
        mul_tensor_3: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 2.0);  mul_tensor_2 = None
        mul_tensor_4: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_3, arg33_1)
        mul_tensor_5: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.2);  mul_tensor_4 = None
        add_tensor: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_tensor_5, arg193_1);  mul_tensor_5 = arg193_1 = None
        mul_tensor_6: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7071067811865476)
        erf_default: "f32[128, 512, 24, 24]" = torch.ops.aten.erf.default(mul_tensor_6);  mul_tensor_6 = None
        add_tensor_1: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_7: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor_1, 0.5);  add_tensor_1 = None
        mul_tensor_8: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, add_tensor)
        mul_tensor_9: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_8, -0.5);  mul_tensor_8 = None
        exp_default: "f32[128, 512, 24, 24]" = torch.ops.aten.exp.default(mul_tensor_9);  mul_tensor_9 = None
        mul_tensor_10: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_11: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_10);  add_tensor = mul_tensor_10 = None
        add_tensor_2: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_11);  mul_tensor_7 = mul_tensor_11 = None
        mul_tensor_12: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_2);  mul_tensor_1 = add_tensor_2 = None
        add_tensor_3: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_785, mul_tensor_12);  mul_785 = mul_tensor_12 = None
        mul_tensor_13: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor_3, 0.2)
        mul_tensor_14: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_13, mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_15: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_13, arg33_1);  mul_tensor_13 = arg33_1 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(mul_tensor_14);  mul_tensor_14 = None
        mul_tensor_16: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_15, 2.0);  mul_tensor_15 = None
        mul_tensor_17: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_16, arg208_1);  mul_tensor_16 = arg208_1 = None
        sum_dim_int_list: "f32[128, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [2, 3], True);  mul_tensor_17 = None
        sub_tensor: "f32[128, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_18: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor);  sigmoid_default = sub_tensor = None
        mul_tensor_19: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_18);  sum_dim_int_list = mul_tensor_18 = None
        sum_dim_int_list_1: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 2, 3]);  mul_tensor_19 = None
        sum_dim_int_list_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_tensor_3, [0, 2, 3]);  add_tensor_3 = None
        return (sum_default, sum_dim_int_list_1, sum_dim_int_list_2)



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
