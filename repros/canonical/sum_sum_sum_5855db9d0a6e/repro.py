"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_training
Pattern hash: 5855db9d0a6e
Shape hash: 639295b6
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_474: "f32[32, 2048, 8, 8]", getitem_226: "f32[32, 384, 8, 8]", getitem_229: "f32[32, 384, 8, 8]", relu_81: "f32[32, 384, 8, 8]", full_default: "f32[]", convolution_81: "f32[32, 384, 8, 8]", unsqueeze_522: "f32[1, 384, 1, 1]", squeeze_244: "f32[384]", primals_492: "f32[384]", convolution_79: "f32[32, 384, 8, 8]", getitem_167: "f32[1, 384, 1, 1]", rsqrt_79: "f32[1, 384, 1, 1]", primals_480: "f32[384]", primals_481: "f32[384]", convolution_78: "f32[32, 384, 8, 8]", getitem_165: "f32[1, 384, 1, 1]", rsqrt_78: "f32[1, 384, 1, 1]", primals_474: "f32[384]", primals_475: "f32[384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:244 in forward, code: return torch.cat(outputs, 1)
        slice_tensor: "f32[32, 768, 8, 8]" = torch.ops.aten.slice.Tensor(add_474, 1, 320, 1088);  add_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        add_tensor: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(getitem_226, getitem_229);  getitem_226 = getitem_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_81, 0);  relu_81 = None
        where_self: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_522);  convolution_81 = unsqueeze_522 = None
        mul_tensor: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00048828125);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00048828125);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_492);  squeeze_244 = primals_492 = None
        unsqueeze_default_6: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:226 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        slice_tensor_1: "f32[32, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_tensor, 1, 0, 384)
        slice_tensor_2: "f32[32, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_tensor, 1, 384, 768);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor_3: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_167)
        mul_tensor_8: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_79);  sub_tensor_3 = None
        unsqueeze_default_9: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_480, -1)
        unsqueeze_default_10: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, -1);  unsqueeze_default_9 = None
        mul_tensor_9: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_8, unsqueeze_default_10);  mul_tensor_8 = unsqueeze_default_10 = None
        unsqueeze_default_11: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_481, -1);  primals_481 = None
        unsqueeze_default_12: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, -1);  unsqueeze_default_11 = None
        add_tensor_1: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_9, unsqueeze_default_12);  mul_tensor_9 = unsqueeze_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        le_scalar_1: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self_1: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar_1, full_default, slice_tensor_2);  le_scalar_1 = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
        unsqueeze_default_13: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_14: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        sum_dim_int_list_2: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_4: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_default_15);  convolution_79 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_4)
        sum_dim_int_list_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 2, 3]);  mul_tensor_10 = None
        mul_tensor_11: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.00048828125);  sum_dim_int_list_2 = None
        unsqueeze_default_16: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_17: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_12: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.00048828125);  sum_dim_int_list_3 = None
        squeeze_dims_1: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
        mul_tensor_13: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_14: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        unsqueeze_default_19: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_20: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_480);  squeeze_dims_1 = primals_480 = None
        unsqueeze_default_22: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_23: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_16: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_21);  sub_tensor_4 = unsqueeze_default_21 = None
        sub_tensor_5: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_16);  where_self_1 = mul_tensor_16 = None
        sub_tensor_6: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_5, unsqueeze_default_18);  sub_tensor_5 = unsqueeze_default_18 = None
        mul_tensor_17: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_24);  sub_tensor_6 = unsqueeze_default_24 = None
        sub_tensor_7: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_165)
        mul_tensor_18: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_7, rsqrt_78);  sub_tensor_7 = None
        unsqueeze_default_25: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_474, -1)
        unsqueeze_default_26: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, -1);  unsqueeze_default_25 = None
        mul_tensor_19: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_18, unsqueeze_default_26);  mul_tensor_18 = unsqueeze_default_26 = None
        unsqueeze_default_27: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_475, -1);  primals_475 = None
        unsqueeze_default_28: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, -1);  unsqueeze_default_27 = None
        add_tensor_2: "f32[32, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_19, unsqueeze_default_28);  mul_tensor_19 = unsqueeze_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[32, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        le_scalar_2: "b8[32, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        where_self_2: "f32[32, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar_2, full_default, slice_tensor_1);  le_scalar_2 = full_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_2: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
        unsqueeze_default_29: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_30: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        sum_dim_int_list_4: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self_2, [0, 2, 3])
        sub_tensor_8: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_default_31);  convolution_78 = unsqueeze_default_31 = None
        mul_tensor_20: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_2, sub_tensor_8)
        sum_dim_int_list_5: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 2, 3]);  mul_tensor_20 = None
        mul_tensor_21: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_4, 0.00048828125);  sum_dim_int_list_4 = None
        unsqueeze_default_32: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_21, 0);  mul_tensor_21 = None
        unsqueeze_default_33: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 2);  unsqueeze_default_32 = None
        unsqueeze_default_34: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_33, 3);  unsqueeze_default_33 = None
        mul_tensor_22: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, 0.00048828125);  sum_dim_int_list_5 = None
        squeeze_dims_3: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
        mul_tensor_23: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_24: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_22, mul_tensor_23);  mul_tensor_22 = mul_tensor_23 = None
        unsqueeze_default_35: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_24, 0);  mul_tensor_24 = None
        unsqueeze_default_36: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_35, 2);  unsqueeze_default_35 = None
        unsqueeze_default_37: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, 3);  unsqueeze_default_36 = None
        mul_tensor_25: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, primals_474);  squeeze_dims_3 = primals_474 = None
        unsqueeze_default_38: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_25, 0);  mul_tensor_25 = None
        unsqueeze_default_39: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 2);  unsqueeze_default_38 = None
        unsqueeze_default_40: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 3);  unsqueeze_default_39 = None
        mul_tensor_26: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_8, unsqueeze_default_37);  sub_tensor_8 = unsqueeze_default_37 = None
        sub_tensor_9: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_2, mul_tensor_26);  where_self_2 = mul_tensor_26 = None
        sub_tensor_10: "f32[32, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_9, unsqueeze_default_34);  sub_tensor_9 = unsqueeze_default_34 = None
        mul_tensor_27: "f32[32, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_10, unsqueeze_default_40);  sub_tensor_10 = unsqueeze_default_40 = None
        return (mul_tensor_7, mul_tensor_17, mul_tensor_27)


def _default_make_inputs():
    return [
    torch.randn(4194304, dtype=torch.float32, device='cuda').as_strided([32, 2048, 8, 8], [131072, 1, 16384, 2048]),  # add_474
    torch.randn(786432, dtype=torch.float32, device='cuda').as_strided([32, 384, 8, 8], [24576, 1, 3072, 384]),  # getitem_226
    torch.randn(786432, dtype=torch.float32, device='cuda').as_strided([32, 384, 8, 8], [24576, 1, 3072, 384]),  # getitem_229
    torch.randn(786432, dtype=torch.float32, device='cuda').as_strided([32, 384, 8, 8], [24576, 1, 3072, 384]),  # relu_81
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(786432, dtype=torch.float32, device='cuda').as_strided([32, 384, 8, 8], [24576, 1, 3072, 384]),  # convolution_81
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(786432, dtype=torch.float32, device='cuda').as_strided([32, 384, 8, 8], [24576, 1, 3072, 384]),  # convolution_79
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(786432, dtype=torch.float32, device='cuda').as_strided([32, 384, 8, 8], [24576, 1, 3072, 384]),  # convolution_78
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
