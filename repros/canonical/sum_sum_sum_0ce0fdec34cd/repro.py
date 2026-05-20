"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 0ce0fdec34cd
Shape hash: 73bc4e05
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 128, 17, 17], f32, stride=(36992, 1, 2176, 128)), T([], f32), T([128, 128, 17, 17], f32, stride=(36992, 1, 2176, 128)), T([128, 128, 17, 17], f32, stride=(36992, 1, 2176, 128)), T([1, 128, 1, 1], f32), T([128], f32), T([128], f32), T([128, 128, 17, 17], f32, stride=(36992, 1, 2176, 128)), T([128, 128, 17, 17], f32, stride=(36992, 1, 2176, 128)), T([128, 128, 17, 17], f32, stride=(36992, 1, 2176, 128)), T([1, 128, 1, 1], f32), T([128], f32), T([128], f32))"

class Repro(torch.nn.Module):
    def forward(self, relu_35: "f32[128, 128, 17, 17]", full_default: "f32[]", getitem_367: "f32[128, 128, 17, 17]", convolution_35: "f32[128, 128, 17, 17]", unsqueeze_1074: "f32[1, 128, 1, 1]", squeeze_106: "f32[128]", primals_216: "f32[128]", relu_32: "f32[128, 128, 17, 17]", getitem_376: "f32[128, 128, 17, 17]", convolution_32: "f32[128, 128, 17, 17]", unsqueeze_1110: "f32[1, 128, 1, 1]", squeeze_97: "f32[128]", primals_198: "f32[128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[128, 128, 17, 17]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_self: "f32[128, 128, 17, 17]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_367);  le_scalar = getitem_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[128, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_1074);  convolution_35 = unsqueeze_1074 = None
        mul_tensor: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.703287197231834e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.703287197231834e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_tensor_4: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_216);  squeeze_106 = primals_216 = None
        unsqueeze_default_6: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar_1: "b8[128, 128, 17, 17]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_self_1: "f32[128, 128, 17, 17]" = torch.ops.aten.where.self(le_scalar_1, full_default, getitem_376);  le_scalar_1 = full_default = getitem_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_3: "f32[128, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1110);  convolution_32 = unsqueeze_1110 = None
        mul_tensor_8: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_3)
        sum_dim_int_list_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.703287197231834e-05);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.703287197231834e-05);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_tensor_12: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_198);  squeeze_97 = primals_198 = None
        unsqueeze_default_15: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[128, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_14);  where_self_1 = mul_tensor_14 = None
        sub_tensor_5: "f32[128, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
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
