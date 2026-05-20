"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_train
Pattern hash: 5fd1e8574dac
Shape hash: fc68ada7
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 64, 112, 112], f32, stride=(802816, 1, 7168, 64)), T([128, 64, 112, 112], f32, stride=(802816, 1, 7168, 64)), T([128, 64, 112, 112], f32, stride=(802816, 1, 7168, 64)), T([], f32), T([128, 64, 112, 112], f32, stride=(802816, 1, 7168, 64)), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([128, 64, 112, 112], f32, stride=(802816, 1, 7168, 64)), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_242: "f32[128, 64, 112, 112]", getitem_245: "f32[128, 64, 112, 112]", relu: "f32[128, 64, 112, 112]", full_default: "f32[]", convolution_1: "f32[128, 64, 112, 112]", unsqueeze_954: "f32[1, 64, 1, 1]", squeeze_4: "f32[64]", primals_12: "f32[64]", convolution: "f32[128, 64, 112, 112]", unsqueeze_966: "f32[1, 64, 1, 1]", squeeze_1: "f32[64]", primals_6: "f32[64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        add_tensor: "f32[128, 64, 112, 112]" = torch.ops.aten.add.Tensor(getitem_242, getitem_245);  getitem_242 = getitem_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_scalar: "b8[128, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_self: "f32[128, 64, 112, 112]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[128, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_954);  convolution_1 = unsqueeze_954 = None
        mul_tensor: "f32[128, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.228077168367346e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.228077168367346e-07);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  squeeze_4 = primals_12 = None
        unsqueeze_default_6: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  mul_tensor_6 = None
        sub_tensor_2: "f32[128, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        sum_dim_int_list_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_3: "f32[128, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_966);  convolution = unsqueeze_966 = None
        mul_tensor_8: "f32[128, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_3)
        sum_dim_int_list_3: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 6.228077168367346e-07);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 6.228077168367346e-07);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_tensor_12: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  squeeze_1 = primals_6 = None
        unsqueeze_default_15: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[128, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[128, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_14);  where_self = mul_tensor_14 = None
        sub_tensor_5: "f32[128, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[128, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        return (mul_tensor_7, mul_tensor_15)


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
