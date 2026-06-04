"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: f48adec0ff36
Shape hash: 2e8d80ed
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 1280, 8, 8], f32, stride=(81920, 1, 10240, 1280)), T([128, 768, 8, 8], i8, stride=(49152, 1, 6144, 768), gen=Index(9)), T([128, 768, 17, 17], f32, stride=(221952, 1, 13056, 768)), T([128, 768, 17, 17], f32, stride=(221952, 1, 13056, 768)), T([128, 192, 17, 17], f32, stride=(55488, 1, 3264, 192)), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([], f32), T([128, 192, 17, 17], f32, stride=(55488, 1, 3264, 192)), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([128, 192, 17, 17], f32, stride=(55488, 1, 3264, 192)), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([128, 192, 17, 17], f32, stride=(55488, 1, 3264, 192)), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), S([98304, 64]), S([98304, 64]), S([128, 768, 17, 17]))"

class Repro(torch.nn.Module):
    def forward(self, add_479: "f32[128, 1280, 8, 8]", getitem_159: "i8[128, 768, 8, 8]", getitem_259: "f32[128, 768, 17, 17]", getitem_265: "f32[128, 768, 17, 17]", convolution_69: "f32[128, 192, 17, 17]", getitem_145: "f32[1, 192, 1, 1]", rsqrt_69: "f32[1, 192, 1, 1]", primals_420: "f32[192]", primals_421: "f32[192]", full_default: "f32[]", convolution_68: "f32[128, 192, 17, 17]", getitem_143: "f32[1, 192, 1, 1]", rsqrt_68: "f32[1, 192, 1, 1]", primals_414: "f32[192]", primals_415: "f32[192]", convolution_63: "f32[128, 192, 17, 17]", getitem_133: "f32[1, 192, 1, 1]", rsqrt_63: "f32[1, 192, 1, 1]", primals_384: "f32[192]", primals_385: "f32[192]", convolution_60: "f32[128, 192, 17, 17]", getitem_127: "f32[1, 192, 1, 1]", rsqrt_60: "f32[1, 192, 1, 1]", primals_366: "f32[192]", primals_367: "f32[192]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:190 in forward, code: return torch.cat(outputs, 1)
        slice_tensor: "f32[128, 768, 8, 8]" = torch.ops.aten.slice.Tensor(add_479, 1, 512, 1280);  add_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:184 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        full_default_1: "f32[98304, 289]" = torch.ops.aten.full.default([98304, 289], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        clone_default: "f32[128, 768, 8, 8]" = torch.ops.aten.clone.default(slice_tensor, memory_format = torch.contiguous_format);  slice_tensor = None
        reshape_default: "f32[98304, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[128, 768, 8, 8]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_159, [3, 3], [17, 17], [2, 2], [0, 0], [1, 1]);  getitem_159 = None
        clone_default_1: "i64[128, 768, 8, 8]" = torch.ops.aten.clone.default(_low_memory_max_pool_offsets_to_indices_default, memory_format = torch.contiguous_format);  _low_memory_max_pool_offsets_to_indices_default = None
        reshape_default_1: "i64[98304, 64]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        scatter_add_default: "f32[98304, 289]" = torch.ops.aten.scatter_add.default(full_default_1, 1, reshape_default_1, reshape_default);  full_default_1 = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[128, 768, 17, 17]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        clone_default_2: "f32[128, 768, 17, 17]" = torch.ops.aten.clone.default(reshape_default_2, memory_format = torch.channels_last);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        add_tensor: "f32[128, 768, 17, 17]" = torch.ops.aten.add.Tensor(clone_default_2, getitem_259);  clone_default_2 = getitem_259 = None
        add_tensor_1: "f32[128, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_tensor, getitem_265);  add_tensor = getitem_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:152 in forward, code: return torch.cat(outputs, 1)
        slice_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 0, 192)
        slice_tensor_2: "f32[128, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 192, 384)
        slice_tensor_3: "f32[128, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 384, 576)
        slice_tensor_4: "f32[128, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 576, 768);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_145)
        mul_tensor: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_69);  sub_tensor = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_420, -1)
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_421, -1);  primals_421 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_2: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        le_scalar: "b8[128, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[128, 192, 17, 17]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor_4);  le_scalar = slice_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
        unsqueeze_default_4: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_default_6);  convolution_69 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.703287197231834e-05);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.703287197231834e-05)
        squeeze_dims_1: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_420);  primals_420 = None
        unsqueeze_default_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        sub_tensor_4: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_143)
        mul_tensor_11: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_68);  sub_tensor_4 = None
        unsqueeze_default_16: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_414, -1)
        unsqueeze_default_17: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        mul_tensor_12: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_11, unsqueeze_default_17);  mul_tensor_11 = unsqueeze_default_17 = None
        unsqueeze_default_18: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_415, -1);  primals_415 = None
        unsqueeze_default_19: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        add_tensor_3: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_12, unsqueeze_default_19);  mul_tensor_12 = unsqueeze_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        le_scalar_1: "b8[128, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        where_self_1: "f32[128, 192, 17, 17]" = torch.ops.aten.where.self(le_scalar_1, full_default, slice_tensor_3);  le_scalar_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_2: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
        unsqueeze_default_20: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_21: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 3);  unsqueeze_default_21 = None
        sum_dim_int_list_2: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_5: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_default_22);  convolution_68 = unsqueeze_default_22 = None
        mul_tensor_13: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_5)
        sum_dim_int_list_3: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 2, 3]);  mul_tensor_13 = None
        mul_tensor_14: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.703287197231834e-05);  sum_dim_int_list_2 = None
        unsqueeze_default_23: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_24: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 3);  unsqueeze_default_24 = None
        mul_tensor_15: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.703287197231834e-05)
        squeeze_dims_3: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
        mul_tensor_16: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_17: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        unsqueeze_default_26: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_17, 0);  mul_tensor_17 = None
        unsqueeze_default_27: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 3);  unsqueeze_default_27 = None
        mul_tensor_18: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, primals_414);  primals_414 = None
        unsqueeze_default_29: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_18, 0);  mul_tensor_18 = None
        unsqueeze_default_30: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        mul_tensor_19: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_28);  sub_tensor_5 = unsqueeze_default_28 = None
        sub_tensor_6: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_19);  where_self_1 = mul_tensor_19 = None
        sub_tensor_7: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_6, unsqueeze_default_25);  sub_tensor_6 = unsqueeze_default_25 = None
        mul_tensor_20: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_31);  sub_tensor_7 = unsqueeze_default_31 = None
        mul_tensor_21: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, squeeze_dims_3);  sum_dim_int_list_3 = squeeze_dims_3 = None
        sub_tensor_8: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_133)
        mul_tensor_22: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_8, rsqrt_63);  sub_tensor_8 = None
        unsqueeze_default_32: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_384, -1)
        unsqueeze_default_33: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, -1);  unsqueeze_default_32 = None
        mul_tensor_23: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_22, unsqueeze_default_33);  mul_tensor_22 = unsqueeze_default_33 = None
        unsqueeze_default_34: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_385, -1);  primals_385 = None
        unsqueeze_default_35: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, -1);  unsqueeze_default_34 = None
        add_tensor_4: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_23, unsqueeze_default_35);  mul_tensor_23 = unsqueeze_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_2: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        le_scalar_2: "b8[128, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_default_2, 0);  relu_default_2 = None
        where_self_2: "f32[128, 192, 17, 17]" = torch.ops.aten.where.self(le_scalar_2, full_default, slice_tensor_2);  le_scalar_2 = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_4: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
        unsqueeze_default_36: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_4, 0);  squeeze_dims_4 = None
        unsqueeze_default_37: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, 2);  unsqueeze_default_36 = None
        unsqueeze_default_38: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_37, 3);  unsqueeze_default_37 = None
        sum_dim_int_list_4: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self_2, [0, 2, 3])
        sub_tensor_9: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_default_38);  convolution_63 = unsqueeze_default_38 = None
        mul_tensor_24: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_self_2, sub_tensor_9)
        sum_dim_int_list_5: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 2, 3]);  mul_tensor_24 = None
        mul_tensor_25: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_4, 2.703287197231834e-05);  sum_dim_int_list_4 = None
        unsqueeze_default_39: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_25, 0);  mul_tensor_25 = None
        unsqueeze_default_40: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 2);  unsqueeze_default_39 = None
        unsqueeze_default_41: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 3);  unsqueeze_default_40 = None
        mul_tensor_26: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, 2.703287197231834e-05)
        squeeze_dims_5: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
        mul_tensor_27: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, squeeze_dims_5)
        mul_tensor_28: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_26, mul_tensor_27);  mul_tensor_26 = mul_tensor_27 = None
        unsqueeze_default_42: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_28, 0);  mul_tensor_28 = None
        unsqueeze_default_43: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 2);  unsqueeze_default_42 = None
        unsqueeze_default_44: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_43, 3);  unsqueeze_default_43 = None
        mul_tensor_29: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, primals_384);  primals_384 = None
        unsqueeze_default_45: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_29, 0);  mul_tensor_29 = None
        unsqueeze_default_46: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_45, 2);  unsqueeze_default_45 = None
        unsqueeze_default_47: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 3);  unsqueeze_default_46 = None
        mul_tensor_30: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_9, unsqueeze_default_44);  sub_tensor_9 = unsqueeze_default_44 = None
        sub_tensor_10: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_self_2, mul_tensor_30);  where_self_2 = mul_tensor_30 = None
        sub_tensor_11: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_10, unsqueeze_default_41);  sub_tensor_10 = unsqueeze_default_41 = None
        mul_tensor_31: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_11, unsqueeze_default_47);  sub_tensor_11 = unsqueeze_default_47 = None
        mul_tensor_32: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, squeeze_dims_5);  sum_dim_int_list_5 = squeeze_dims_5 = None
        sub_tensor_12: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_127)
        mul_tensor_33: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_12, rsqrt_60);  sub_tensor_12 = None
        unsqueeze_default_48: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_366, -1)
        unsqueeze_default_49: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, -1);  unsqueeze_default_48 = None
        mul_tensor_34: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_33, unsqueeze_default_49);  mul_tensor_33 = unsqueeze_default_49 = None
        unsqueeze_default_50: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_367, -1);  primals_367 = None
        unsqueeze_default_51: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_50, -1);  unsqueeze_default_50 = None
        add_tensor_5: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_34, unsqueeze_default_51);  mul_tensor_34 = unsqueeze_default_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_3: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_5);  add_tensor_5 = None
        le_scalar_3: "b8[128, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_default_3, 0);  relu_default_3 = None
        where_self_3: "f32[128, 192, 17, 17]" = torch.ops.aten.where.self(le_scalar_3, full_default, slice_tensor_1);  le_scalar_3 = full_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_6: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
        unsqueeze_default_52: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_6, 0);  squeeze_dims_6 = None
        unsqueeze_default_53: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_52, 2);  unsqueeze_default_52 = None
        unsqueeze_default_54: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_53, 3);  unsqueeze_default_53 = None
        sum_dim_int_list_6: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self_3, [0, 2, 3])
        sub_tensor_13: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_default_54);  convolution_60 = unsqueeze_default_54 = None
        mul_tensor_35: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_self_3, sub_tensor_13)
        sum_dim_int_list_7: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 2, 3]);  mul_tensor_35 = None
        mul_tensor_36: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_6, 2.703287197231834e-05);  sum_dim_int_list_6 = None
        unsqueeze_default_55: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_36, 0);  mul_tensor_36 = None
        unsqueeze_default_56: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_55, 2);  unsqueeze_default_55 = None
        unsqueeze_default_57: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_56, 3);  unsqueeze_default_56 = None
        mul_tensor_37: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_7, 2.703287197231834e-05)
        squeeze_dims_7: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_tensor_38: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, squeeze_dims_7)
        mul_tensor_39: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_37, mul_tensor_38);  mul_tensor_37 = mul_tensor_38 = None
        unsqueeze_default_58: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_39, 0);  mul_tensor_39 = None
        unsqueeze_default_59: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_58, 2);  unsqueeze_default_58 = None
        unsqueeze_default_60: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_59, 3);  unsqueeze_default_59 = None
        mul_tensor_40: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, primals_366);  primals_366 = None
        unsqueeze_default_61: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_40, 0);  mul_tensor_40 = None
        unsqueeze_default_62: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_61, 2);  unsqueeze_default_61 = None
        unsqueeze_default_63: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_62, 3);  unsqueeze_default_62 = None
        mul_tensor_41: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_13, unsqueeze_default_60);  sub_tensor_13 = unsqueeze_default_60 = None
        sub_tensor_14: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_self_3, mul_tensor_41);  where_self_3 = mul_tensor_41 = None
        sub_tensor_15: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_14, unsqueeze_default_57);  sub_tensor_14 = unsqueeze_default_57 = None
        mul_tensor_42: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_15, unsqueeze_default_63);  sub_tensor_15 = unsqueeze_default_63 = None
        mul_tensor_43: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_7, squeeze_dims_7);  sum_dim_int_list_7 = squeeze_dims_7 = None
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
