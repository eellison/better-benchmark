"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 3cd8c07ebace
Shape hash: 0d266232
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 768, 17, 17], f32, stride=(221952, 1, 13056, 768)), T([128, 288, 17, 17], i8, stride=(83232, 1, 4896, 288), gen=Index(9)), T([128, 288, 35, 35], f32, stride=(352800, 1, 10080, 288)), T([128, 288, 35, 35], f32, stride=(352800, 1, 10080, 288)), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([], f32), T([128, 96, 35, 35], f32, stride=(117600, 1, 3360, 96)), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([96], f32), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), S([36864, 289]), S([36864, 289]), S([128, 288, 35, 35]))"

class Repro(torch.nn.Module):
    def forward(self, add_493: "f32[128, 768, 17, 17]", getitem_65: "i8[128, 288, 17, 17]", getitem_394: "f32[128, 288, 35, 35]", getitem_397: "f32[128, 288, 35, 35]", convolution_25: "f32[128, 64, 35, 35]", getitem_55: "f32[1, 64, 1, 1]", rsqrt_25: "f32[1, 64, 1, 1]", primals_156: "f32[64]", primals_157: "f32[64]", full_default: "f32[]", convolution_24: "f32[128, 96, 35, 35]", getitem_53: "f32[1, 96, 1, 1]", rsqrt_24: "f32[1, 96, 1, 1]", primals_150: "f32[96]", primals_151: "f32[96]", convolution_21: "f32[128, 64, 35, 35]", getitem_47: "f32[1, 64, 1, 1]", rsqrt_21: "f32[1, 64, 1, 1]", primals_132: "f32[64]", primals_133: "f32[64]", convolution_19: "f32[128, 64, 35, 35]", getitem_43: "f32[1, 64, 1, 1]", rsqrt_19: "f32[1, 64, 1, 1]", primals_120: "f32[64]", primals_121: "f32[64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:100 in forward, code: return torch.cat(outputs, 1)
        slice_tensor: "f32[128, 288, 17, 17]" = torch.ops.aten.slice.Tensor(add_493, 1, 480, 768);  add_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:93 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        full_default_1: "f32[36864, 1225]" = torch.ops.aten.full.default([36864, 1225], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        clone_default: "f32[128, 288, 17, 17]" = torch.ops.aten.clone.default(slice_tensor, memory_format = torch.contiguous_format);  slice_tensor = None
        reshape_default: "f32[36864, 289]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[128, 288, 17, 17]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_65, [3, 3], [35, 35], [2, 2], [0, 0], [1, 1]);  getitem_65 = None
        clone_default_1: "i64[128, 288, 17, 17]" = torch.ops.aten.clone.default(_low_memory_max_pool_offsets_to_indices_default, memory_format = torch.contiguous_format);  _low_memory_max_pool_offsets_to_indices_default = None
        reshape_default_1: "i64[36864, 289]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        scatter_add_default: "f32[36864, 1225]" = torch.ops.aten.scatter_add.default(full_default_1, 1, reshape_default_1, reshape_default);  full_default_1 = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[128, 288, 35, 35]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        clone_default_2: "f32[128, 288, 35, 35]" = torch.ops.aten.clone.default(reshape_default_2, memory_format = torch.channels_last);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        add_tensor: "f32[128, 288, 35, 35]" = torch.ops.aten.add.Tensor(clone_default_2, getitem_394);  clone_default_2 = getitem_394 = None
        add_tensor_1: "f32[128, 288, 35, 35]" = torch.ops.aten.add.Tensor(add_tensor, getitem_397);  add_tensor = getitem_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:65 in forward, code: return torch.cat(outputs, 1)
        slice_tensor_1: "f32[128, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 0, 64)
        slice_tensor_2: "f32[128, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 64, 128)
        slice_tensor_3: "f32[128, 96, 35, 35]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 128, 224)
        slice_tensor_4: "f32[128, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 224, 288);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_55)
        mul_tensor: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_25);  sub_tensor = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_2: "f32[128, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 64, 35, 35]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        le_scalar: "b8[128, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[128, 64, 35, 35]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor_4);  le_scalar = slice_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        unsqueeze_default_4: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_default_6);  convolution_25 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.3775510204081635e-06);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.3775510204081635e-06)
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_156);  primals_156 = None
        unsqueeze_default_13: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        sub_tensor_4: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_53)
        mul_tensor_11: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_24);  sub_tensor_4 = None
        unsqueeze_default_16: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_default_17: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        mul_tensor_12: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor_11, unsqueeze_default_17);  mul_tensor_11 = unsqueeze_default_17 = None
        unsqueeze_default_18: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_151, -1);  primals_151 = None
        unsqueeze_default_19: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        add_tensor_3: "f32[128, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_12, unsqueeze_default_19);  mul_tensor_12 = unsqueeze_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 96, 35, 35]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        le_scalar_1: "b8[128, 96, 35, 35]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        where_self_1: "f32[128, 96, 35, 35]" = torch.ops.aten.where.self(le_scalar_1, full_default, slice_tensor_3);  le_scalar_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_2: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        unsqueeze_default_20: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_21: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 3);  unsqueeze_default_21 = None
        sum_dim_int_list_2: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_5: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_default_22);  convolution_24 = unsqueeze_default_22 = None
        mul_tensor_13: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_5)
        sum_dim_int_list_3: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 2, 3]);  mul_tensor_13 = None
        mul_tensor_14: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 6.3775510204081635e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_23: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_24: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 3);  unsqueeze_default_24 = None
        mul_tensor_15: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 6.3775510204081635e-06)
        squeeze_dims_3: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_tensor_16: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_17: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        unsqueeze_default_26: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_17, 0);  mul_tensor_17 = None
        unsqueeze_default_27: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 3);  unsqueeze_default_27 = None
        mul_tensor_18: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, primals_150);  primals_150 = None
        unsqueeze_default_29: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_tensor_18, 0);  mul_tensor_18 = None
        unsqueeze_default_30: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        mul_tensor_19: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_28);  sub_tensor_5 = unsqueeze_default_28 = None
        sub_tensor_6: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_19);  where_self_1 = mul_tensor_19 = None
        sub_tensor_7: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_tensor_6, unsqueeze_default_25);  sub_tensor_6 = unsqueeze_default_25 = None
        mul_tensor_20: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_31);  sub_tensor_7 = unsqueeze_default_31 = None
        mul_tensor_21: "f32[96]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, squeeze_dims_3);  sum_dim_int_list_3 = squeeze_dims_3 = None
        sub_tensor_8: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_47)
        mul_tensor_22: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_8, rsqrt_21);  sub_tensor_8 = None
        unsqueeze_default_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_default_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, -1);  unsqueeze_default_32 = None
        mul_tensor_23: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor_22, unsqueeze_default_33);  mul_tensor_22 = unsqueeze_default_33 = None
        unsqueeze_default_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_133, -1);  primals_133 = None
        unsqueeze_default_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, -1);  unsqueeze_default_34 = None
        add_tensor_4: "f32[128, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_23, unsqueeze_default_35);  mul_tensor_23 = unsqueeze_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_2: "f32[128, 64, 35, 35]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        le_scalar_2: "b8[128, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_default_2, 0);  relu_default_2 = None
        where_self_2: "f32[128, 64, 35, 35]" = torch.ops.aten.where.self(le_scalar_2, full_default, slice_tensor_2);  le_scalar_2 = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_4: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        unsqueeze_default_36: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_4, 0);  squeeze_dims_4 = None
        unsqueeze_default_37: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, 2);  unsqueeze_default_36 = None
        unsqueeze_default_38: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_37, 3);  unsqueeze_default_37 = None
        sum_dim_int_list_4: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self_2, [0, 2, 3])
        sub_tensor_9: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_default_38);  convolution_21 = unsqueeze_default_38 = None
        mul_tensor_24: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_self_2, sub_tensor_9)
        sum_dim_int_list_5: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 2, 3]);  mul_tensor_24 = None
        mul_tensor_25: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_4, 6.3775510204081635e-06);  sum_dim_int_list_4 = None
        unsqueeze_default_39: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_25, 0);  mul_tensor_25 = None
        unsqueeze_default_40: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 2);  unsqueeze_default_39 = None
        unsqueeze_default_41: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 3);  unsqueeze_default_40 = None
        mul_tensor_26: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, 6.3775510204081635e-06)
        squeeze_dims_5: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_tensor_27: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, squeeze_dims_5)
        mul_tensor_28: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_26, mul_tensor_27);  mul_tensor_26 = mul_tensor_27 = None
        unsqueeze_default_42: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_28, 0);  mul_tensor_28 = None
        unsqueeze_default_43: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 2);  unsqueeze_default_42 = None
        unsqueeze_default_44: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_43, 3);  unsqueeze_default_43 = None
        mul_tensor_29: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, primals_132);  primals_132 = None
        unsqueeze_default_45: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_29, 0);  mul_tensor_29 = None
        unsqueeze_default_46: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_45, 2);  unsqueeze_default_45 = None
        unsqueeze_default_47: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 3);  unsqueeze_default_46 = None
        mul_tensor_30: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_9, unsqueeze_default_44);  sub_tensor_9 = unsqueeze_default_44 = None
        sub_tensor_10: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_self_2, mul_tensor_30);  where_self_2 = mul_tensor_30 = None
        sub_tensor_11: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_tensor_10, unsqueeze_default_41);  sub_tensor_10 = unsqueeze_default_41 = None
        mul_tensor_31: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_11, unsqueeze_default_47);  sub_tensor_11 = unsqueeze_default_47 = None
        mul_tensor_32: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, squeeze_dims_5);  sum_dim_int_list_5 = squeeze_dims_5 = None
        sub_tensor_12: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_43)
        mul_tensor_33: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_12, rsqrt_19);  sub_tensor_12 = None
        unsqueeze_default_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_default_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, -1);  unsqueeze_default_48 = None
        mul_tensor_34: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor_33, unsqueeze_default_49);  mul_tensor_33 = unsqueeze_default_49 = None
        unsqueeze_default_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
        unsqueeze_default_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_50, -1);  unsqueeze_default_50 = None
        add_tensor_5: "f32[128, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_34, unsqueeze_default_51);  mul_tensor_34 = unsqueeze_default_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_3: "f32[128, 64, 35, 35]" = torch.ops.aten.relu.default(add_tensor_5);  add_tensor_5 = None
        le_scalar_3: "b8[128, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_default_3, 0);  relu_default_3 = None
        where_self_3: "f32[128, 64, 35, 35]" = torch.ops.aten.where.self(le_scalar_3, full_default, slice_tensor_1);  le_scalar_3 = full_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        unsqueeze_default_52: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_6, 0);  squeeze_dims_6 = None
        unsqueeze_default_53: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_52, 2);  unsqueeze_default_52 = None
        unsqueeze_default_54: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_53, 3);  unsqueeze_default_53 = None
        sum_dim_int_list_6: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self_3, [0, 2, 3])
        sub_tensor_13: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_default_54);  convolution_19 = unsqueeze_default_54 = None
        mul_tensor_35: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_self_3, sub_tensor_13)
        sum_dim_int_list_7: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 2, 3]);  mul_tensor_35 = None
        mul_tensor_36: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_6, 6.3775510204081635e-06);  sum_dim_int_list_6 = None
        unsqueeze_default_55: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_36, 0);  mul_tensor_36 = None
        unsqueeze_default_56: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_55, 2);  unsqueeze_default_55 = None
        unsqueeze_default_57: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_56, 3);  unsqueeze_default_56 = None
        mul_tensor_37: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_7, 6.3775510204081635e-06)
        squeeze_dims_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_tensor_38: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, squeeze_dims_7)
        mul_tensor_39: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_37, mul_tensor_38);  mul_tensor_37 = mul_tensor_38 = None
        unsqueeze_default_58: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_39, 0);  mul_tensor_39 = None
        unsqueeze_default_59: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_58, 2);  unsqueeze_default_58 = None
        unsqueeze_default_60: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_59, 3);  unsqueeze_default_59 = None
        mul_tensor_40: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, primals_120);  primals_120 = None
        unsqueeze_default_61: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_40, 0);  mul_tensor_40 = None
        unsqueeze_default_62: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_61, 2);  unsqueeze_default_61 = None
        unsqueeze_default_63: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_62, 3);  unsqueeze_default_62 = None
        mul_tensor_41: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_13, unsqueeze_default_60);  sub_tensor_13 = unsqueeze_default_60 = None
        sub_tensor_14: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_self_3, mul_tensor_41);  where_self_3 = mul_tensor_41 = None
        sub_tensor_15: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_tensor_14, unsqueeze_default_57);  sub_tensor_14 = unsqueeze_default_57 = None
        mul_tensor_42: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_15, unsqueeze_default_63);  sub_tensor_15 = unsqueeze_default_63 = None
        mul_tensor_43: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_7, squeeze_dims_7);  sum_dim_int_list_7 = squeeze_dims_7 = None
        return (mul_tensor_9, mul_tensor_10, mul_tensor_20, mul_tensor_21, mul_tensor_31, mul_tensor_32, mul_tensor_42, mul_tensor_43)

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
