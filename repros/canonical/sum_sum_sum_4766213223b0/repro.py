"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 4766213223b0
Shape hash: 9a32afb1
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
_shapes_config = "(T([128, 2048, 8, 8], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32))"

class Repro(torch.nn.Module):
    def forward(self, div: "f32[128, 2048, 8, 8]", convolution_88: "f32[128, 384, 8, 8]", getitem_185: "f32[1, 384, 1, 1]", rsqrt_88: "f32[1, 384, 1, 1]", primals_534: "f32[384]", primals_535: "f32[384]", full_default: "f32[]", convolution_87: "f32[128, 384, 8, 8]", getitem_183: "f32[1, 384, 1, 1]", rsqrt_87: "f32[1, 384, 1, 1]", primals_528: "f32[384]", primals_529: "f32[384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:244 in forward, code: return torch.cat(outputs, 1)
        slice_tensor: "f32[128, 768, 8, 8]" = torch.ops.aten.slice.Tensor(div, 1, 320, 1088);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:226 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        slice_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_tensor, 1, 0, 384)
        slice_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_tensor, 1, 384, 768);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_185)
        mul_tensor: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_88);  sub_tensor = None
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_534, -1)
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_535, -1);  primals_535 = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[128, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor_2);  le_scalar = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3]);  getitem_185 = None
        unsqueeze_default_4: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_default_6);  convolution_88 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0001220703125);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0001220703125);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_534);  squeeze_dims_1 = primals_534 = None
        unsqueeze_default_13: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        sub_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_183)
        mul_tensor_10: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_87);  sub_tensor_4 = None
        unsqueeze_default_16: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_528, -1)
        unsqueeze_default_17: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        mul_tensor_11: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_10, unsqueeze_default_17);  mul_tensor_10 = unsqueeze_default_17 = None
        unsqueeze_default_18: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_529, -1);  primals_529 = None
        unsqueeze_default_19: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        add_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_11, unsqueeze_default_19);  mul_tensor_11 = unsqueeze_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        le_scalar_1: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        where_self_1: "f32[128, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar_1, full_default, slice_tensor_1);  le_scalar_1 = full_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_2: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
        unsqueeze_default_20: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_21: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 3);  unsqueeze_default_21 = None
        sum_dim_int_list_2: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_5: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_default_22);  convolution_87 = unsqueeze_default_22 = None
        mul_tensor_12: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_5)
        sum_dim_int_list_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 2, 3]);  mul_tensor_12 = None
        mul_tensor_13: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.0001220703125);  sum_dim_int_list_2 = None
        unsqueeze_default_23: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_24: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 3);  unsqueeze_default_24 = None
        mul_tensor_14: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.0001220703125);  sum_dim_int_list_3 = None
        squeeze_dims_3: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_16: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_14, mul_tensor_15);  mul_tensor_14 = mul_tensor_15 = None
        unsqueeze_default_26: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_16, 0);  mul_tensor_16 = None
        unsqueeze_default_27: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 3);  unsqueeze_default_27 = None
        mul_tensor_17: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, primals_528);  squeeze_dims_3 = primals_528 = None
        unsqueeze_default_29: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_17, 0);  mul_tensor_17 = None
        unsqueeze_default_30: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        mul_tensor_18: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_28);  sub_tensor_5 = unsqueeze_default_28 = None
        sub_tensor_6: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_18);  where_self_1 = mul_tensor_18 = None
        sub_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_6, unsqueeze_default_25);  sub_tensor_6 = unsqueeze_default_25 = None
        mul_tensor_19: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_31);  sub_tensor_7 = unsqueeze_default_31 = None
        return (mul_tensor_9, mul_tensor_19)



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
