"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: 4d7e4d27796e
Shape hash: 455f8b04
"""
_shapes_config = "(T([128, 288, 35, 35], f32, stride=(352800, 1, 10080, 288)), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([], f32), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([128, 48, 35, 35], f32, stride=(58800, 1, 1680, 48)), T([128, 48, 35, 35], f32, stride=(58800, 1, 1680, 48)), T([128, 48, 35, 35], f32, stride=(58800, 1, 1680, 48)), T([1, 48, 1, 1], f32), T([48], f32), T([48], f32), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, add_498: "f32[128, 288, 35, 35]", convolution_18: "f32[128, 64, 35, 35]", getitem_41: "f32[1, 64, 1, 1]", rsqrt_18: "f32[1, 64, 1, 1]", primals_114: "f32[64]", primals_115: "f32[64]", full_default: "f32[]", relu_15: "f32[128, 64, 35, 35]", getitem_427: "f32[128, 64, 35, 35]", convolution_15: "f32[128, 64, 35, 35]", unsqueeze_1314: "f32[1, 64, 1, 1]", squeeze_46: "f32[64]", primals_96: "f32[64]", relu_13: "f32[128, 48, 35, 35]", getitem_433: "f32[128, 48, 35, 35]", convolution_13: "f32[128, 48, 35, 35]", unsqueeze_1338: "f32[1, 48, 1, 1]", squeeze_40: "f32[48]", primals_84: "f32[48]", convolution_12: "f32[128, 64, 35, 35]", getitem_29: "f32[1, 64, 1, 1]", rsqrt_12: "f32[1, 64, 1, 1]", primals_78: "f32[64]", primals_79: "f32[64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:65 in forward, code: return torch.cat(outputs, 1)
        slice_tensor: "f32[128, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_498, 1, 0, 64)
        slice_tensor_1: "f32[128, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_498, 1, 224, 288);  add_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_41)
        mul_tensor: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_18);  sub_tensor = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 64, 35, 35]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[128, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[128, 64, 35, 35]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor_1);  le_scalar = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        unsqueeze_default_4: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_default_6);  convolution_18 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.3775510204081635e-06);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.3775510204081635e-06);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_114);  squeeze_dims_1 = primals_114 = None
        unsqueeze_default_13: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar_1: "b8[128, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_self_1: "f32[128, 64, 35, 35]" = torch.ops.aten.where.self(le_scalar_1, full_default, getitem_427);  le_scalar_1 = getitem_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_4: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1314);  convolution_15 = unsqueeze_1314 = None
        mul_tensor_10: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_4)
        sum_dim_int_list_3: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 2, 3]);  mul_tensor_10 = None
        mul_tensor_11: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 6.3775510204081635e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_16: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_17: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_12: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 6.3775510204081635e-06);  sum_dim_int_list_3 = None
        mul_tensor_13: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_tensor_14: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        unsqueeze_default_19: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_20: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  squeeze_46 = primals_96 = None
        unsqueeze_default_22: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_23: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_16: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_21);  sub_tensor_4 = unsqueeze_default_21 = None
        sub_tensor_5: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_16);  where_self_1 = mul_tensor_16 = None
        sub_tensor_6: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_tensor_5, unsqueeze_default_18);  sub_tensor_5 = unsqueeze_default_18 = None
        mul_tensor_17: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_24);  sub_tensor_6 = unsqueeze_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar_2: "b8[128, 48, 35, 35]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_self_2: "f32[128, 48, 35, 35]" = torch.ops.aten.where.self(le_scalar_2, full_default, getitem_433);  le_scalar_2 = getitem_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list_4: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_self_2, [0, 2, 3])
        sub_tensor_7: "f32[128, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1338);  convolution_13 = unsqueeze_1338 = None
        mul_tensor_18: "f32[128, 48, 35, 35]" = torch.ops.aten.mul.Tensor(where_self_2, sub_tensor_7)
        sum_dim_int_list_5: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 2, 3]);  mul_tensor_18 = None
        mul_tensor_19: "f32[48]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_4, 6.3775510204081635e-06);  sum_dim_int_list_4 = None
        unsqueeze_default_25: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_tensor_19, 0);  mul_tensor_19 = None
        unsqueeze_default_26: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 2);  unsqueeze_default_25 = None
        unsqueeze_default_27: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 3);  unsqueeze_default_26 = None
        mul_tensor_20: "f32[48]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, 6.3775510204081635e-06);  sum_dim_int_list_5 = None
        mul_tensor_21: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_tensor_22: "f32[48]" = torch.ops.aten.mul.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        unsqueeze_default_28: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_tensor_22, 0);  mul_tensor_22 = None
        unsqueeze_default_29: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 2);  unsqueeze_default_28 = None
        unsqueeze_default_30: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 3);  unsqueeze_default_29 = None
        mul_tensor_23: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  squeeze_40 = primals_84 = None
        unsqueeze_default_31: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_tensor_23, 0);  mul_tensor_23 = None
        unsqueeze_default_32: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_31, 2);  unsqueeze_default_31 = None
        unsqueeze_default_33: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 3);  unsqueeze_default_32 = None
        mul_tensor_24: "f32[128, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_30);  sub_tensor_7 = unsqueeze_default_30 = None
        sub_tensor_8: "f32[128, 48, 35, 35]" = torch.ops.aten.sub.Tensor(where_self_2, mul_tensor_24);  where_self_2 = mul_tensor_24 = None
        sub_tensor_9: "f32[128, 48, 35, 35]" = torch.ops.aten.sub.Tensor(sub_tensor_8, unsqueeze_default_27);  sub_tensor_8 = unsqueeze_default_27 = None
        mul_tensor_25: "f32[128, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_9, unsqueeze_default_33);  sub_tensor_9 = unsqueeze_default_33 = None
        sub_tensor_10: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_29)
        mul_tensor_26: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_10, rsqrt_12);  sub_tensor_10 = None
        unsqueeze_default_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_default_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, -1);  unsqueeze_default_34 = None
        mul_tensor_27: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor_26, unsqueeze_default_35);  mul_tensor_26 = unsqueeze_default_35 = None
        unsqueeze_default_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_default_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, -1);  unsqueeze_default_36 = None
        add_tensor_1: "f32[128, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_27, unsqueeze_default_37);  mul_tensor_27 = unsqueeze_default_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 64, 35, 35]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        le_scalar_3: "b8[128, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        where_self_3: "f32[128, 64, 35, 35]" = torch.ops.aten.where.self(le_scalar_3, full_default, slice_tensor);  le_scalar_3 = full_default = slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        unsqueeze_default_38: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_39: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 2);  unsqueeze_default_38 = None
        unsqueeze_default_40: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 3);  unsqueeze_default_39 = None
        sum_dim_int_list_6: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self_3, [0, 2, 3])
        sub_tensor_11: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_default_40);  convolution_12 = unsqueeze_default_40 = None
        mul_tensor_28: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_self_3, sub_tensor_11)
        sum_dim_int_list_7: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [0, 2, 3]);  mul_tensor_28 = None
        mul_tensor_29: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_6, 6.3775510204081635e-06);  sum_dim_int_list_6 = None
        unsqueeze_default_41: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_29, 0);  mul_tensor_29 = None
        unsqueeze_default_42: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_41, 2);  unsqueeze_default_41 = None
        unsqueeze_default_43: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 3);  unsqueeze_default_42 = None
        mul_tensor_30: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_7, 6.3775510204081635e-06);  sum_dim_int_list_7 = None
        squeeze_dims_3: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_tensor_31: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_32: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        unsqueeze_default_44: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_32, 0);  mul_tensor_32 = None
        unsqueeze_default_45: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, 2);  unsqueeze_default_44 = None
        unsqueeze_default_46: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_45, 3);  unsqueeze_default_45 = None
        mul_tensor_33: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, primals_78);  squeeze_dims_3 = primals_78 = None
        unsqueeze_default_47: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_33, 0);  mul_tensor_33 = None
        unsqueeze_default_48: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_47, 2);  unsqueeze_default_47 = None
        unsqueeze_default_49: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, 3);  unsqueeze_default_48 = None
        mul_tensor_34: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_11, unsqueeze_default_46);  sub_tensor_11 = unsqueeze_default_46 = None
        sub_tensor_12: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_self_3, mul_tensor_34);  where_self_3 = mul_tensor_34 = None
        sub_tensor_13: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_tensor_12, unsqueeze_default_43);  sub_tensor_12 = unsqueeze_default_43 = None
        mul_tensor_35: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_13, unsqueeze_default_49);  sub_tensor_13 = unsqueeze_default_49 = None
        return (mul_tensor_9, mul_tensor_17, mul_tensor_25, mul_tensor_35)



def make_inputs():
    return [
    torch.randn(45158400, dtype=torch.float32, device='cuda').as_strided([128, 288, 35, 35], [352800, 1, 10080, 288]),  # add_498
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([128, 64, 35, 35], [78400, 1, 2240, 64]),  # convolution_18
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([128, 64, 35, 35], [78400, 1, 2240, 64]),  # relu_15
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([128, 64, 35, 35], [78400, 1, 2240, 64]),  # getitem_427
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([128, 64, 35, 35], [78400, 1, 2240, 64]),  # convolution_15
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn(7526400, dtype=torch.float32, device='cuda').as_strided([128, 48, 35, 35], [58800, 1, 1680, 48]),  # relu_13
    torch.randn(7526400, dtype=torch.float32, device='cuda').as_strided([128, 48, 35, 35], [58800, 1, 1680, 48]),  # getitem_433
    torch.randn(7526400, dtype=torch.float32, device='cuda').as_strided([128, 48, 35, 35], [58800, 1, 1680, 48]),  # convolution_13
    torch.randn([1, 48, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48], dtype=torch.float32, device='cuda'),
    torch.randn([48], dtype=torch.float32, device='cuda'),
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([128, 64, 35, 35], [78400, 1, 2240, 64]),  # convolution_12
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
