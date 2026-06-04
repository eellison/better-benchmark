"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 92327e661e73
Shape hash: c67ecfd4
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 2048], f32), T([128, 192, 8, 8], f32, stride=(12288, 1, 1536, 192)), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 320, 8, 8], f32, stride=(20480, 1, 2560, 320)), T([1, 320, 1, 1], f32), T([1, 320, 1, 1], f32), T([320], f32), T([320], f32), S([128, 2048, 1, 1]), S([128, 2048, 8, 8]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 2048]", convolution_93: "f32[128, 192, 8, 8]", getitem_195: "f32[1, 192, 1, 1]", rsqrt_93: "f32[1, 192, 1, 1]", primals_564: "f32[192]", primals_565: "f32[192]", convolution_92: "f32[128, 384, 8, 8]", getitem_193: "f32[1, 384, 1, 1]", rsqrt_92: "f32[1, 384, 1, 1]", primals_558: "f32[384]", primals_559: "f32[384]", convolution_91: "f32[128, 384, 8, 8]", getitem_191: "f32[1, 384, 1, 1]", rsqrt_91: "f32[1, 384, 1, 1]", primals_552: "f32[384]", primals_553: "f32[384]", convolution_88: "f32[128, 384, 8, 8]", getitem_185: "f32[1, 384, 1, 1]", rsqrt_88: "f32[1, 384, 1, 1]", primals_534: "f32[384]", primals_535: "f32[384]", convolution_87: "f32[128, 384, 8, 8]", getitem_183: "f32[1, 384, 1, 1]", rsqrt_87: "f32[1, 384, 1, 1]", primals_528: "f32[384]", primals_529: "f32[384]", convolution_85: "f32[128, 320, 8, 8]", getitem_179: "f32[1, 320, 1, 1]", rsqrt_85: "f32[1, 320, 1, 1]", primals_516: "f32[320]", primals_517: "f32[320]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 2048, 1, 1]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_dim: "f32[128, 2048, 1]" = torch.ops.aten.squeeze.dim(reshape_default, 3);  reshape_default = None
        squeeze_dim_1: "f32[128, 2048]" = torch.ops.aten.squeeze.dim(squeeze_dim, 2);  squeeze_dim = None
        full_default: "f32[262144]" = torch.ops.aten.full.default([262144], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter_default: "f32[262144]" = torch.ops.aten.as_strided_scatter.default(full_default, squeeze_dim_1, [128, 2048], [2048, 1], 0);  full_default = squeeze_dim_1 = None
        as_strided_default: "f32[128, 2048, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter_default, [128, 2048, 1, 1], [2048, 1, 1, 1], 0);  as_strided_scatter_default = None
        expand_default: "f32[128, 2048, 8, 8]" = torch.ops.aten.expand.default(as_strided_default, _shape_param_1);  as_strided_default = _shape_param_1 = None
        div_scalar: "f32[128, 2048, 8, 8]" = torch.ops.aten.div.Scalar(expand_default, 64);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:244 in forward, code: return torch.cat(outputs, 1)
        slice_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.slice.Tensor(div_scalar, 1, 0, 320)
        slice_tensor_1: "f32[128, 768, 8, 8]" = torch.ops.aten.slice.Tensor(div_scalar, 1, 320, 1088)
        slice_tensor_2: "f32[128, 768, 8, 8]" = torch.ops.aten.slice.Tensor(div_scalar, 1, 1088, 1856)
        slice_tensor_3: "f32[128, 192, 8, 8]" = torch.ops.aten.slice.Tensor(div_scalar, 1, 1856, 2048);  div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_195)
        mul_tensor: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_93);  sub_tensor = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_564, -1)
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_565, -1);  primals_565 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 192, 8, 8]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[128, 192, 8, 8]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 192, 8, 8]" = torch.ops.aten.where.self(le_scalar, full_default_1, slice_tensor_3);  le_scalar = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
        unsqueeze_default_4: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_default_6);  convolution_93 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0001220703125);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0001220703125)
        squeeze_dims_1: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_564);  primals_564 = None
        unsqueeze_default_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:234 in _forward, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
        slice_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 1, 0, 384)
        slice_tensor_5: "f32[128, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 1, 384, 768);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_193)
        mul_tensor_11: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_92);  sub_tensor_4 = None
        unsqueeze_default_16: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_558, -1)
        unsqueeze_default_17: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        mul_tensor_12: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_11, unsqueeze_default_17);  mul_tensor_11 = unsqueeze_default_17 = None
        unsqueeze_default_18: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_559, -1);  primals_559 = None
        unsqueeze_default_19: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        add_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_12, unsqueeze_default_19);  mul_tensor_12 = unsqueeze_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        le_scalar_1: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        where_self_1: "f32[128, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar_1, full_default_1, slice_tensor_5);  le_scalar_1 = slice_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_2: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
        unsqueeze_default_20: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_21: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 3);  unsqueeze_default_21 = None
        sum_dim_int_list_2: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_5: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_default_22);  convolution_92 = unsqueeze_default_22 = None
        mul_tensor_13: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_5)
        sum_dim_int_list_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 2, 3]);  mul_tensor_13 = None
        mul_tensor_14: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.0001220703125);  sum_dim_int_list_2 = None
        unsqueeze_default_23: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_24: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 3);  unsqueeze_default_24 = None
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.0001220703125)
        squeeze_dims_3: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
        mul_tensor_16: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_17: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        unsqueeze_default_26: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_17, 0);  mul_tensor_17 = None
        unsqueeze_default_27: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 3);  unsqueeze_default_27 = None
        mul_tensor_18: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, primals_558);  primals_558 = None
        unsqueeze_default_29: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_18, 0);  mul_tensor_18 = None
        unsqueeze_default_30: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        mul_tensor_19: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_28);  sub_tensor_5 = unsqueeze_default_28 = None
        sub_tensor_6: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_19);  where_self_1 = mul_tensor_19 = None
        sub_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_6, unsqueeze_default_25);  sub_tensor_6 = unsqueeze_default_25 = None
        mul_tensor_20: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_31);  sub_tensor_7 = unsqueeze_default_31 = None
        mul_tensor_21: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, squeeze_dims_3);  sum_dim_int_list_3 = squeeze_dims_3 = None
        sub_tensor_8: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_91, getitem_191)
        mul_tensor_22: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_8, rsqrt_91);  sub_tensor_8 = None
        unsqueeze_default_32: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_552, -1)
        unsqueeze_default_33: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, -1);  unsqueeze_default_32 = None
        mul_tensor_23: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_22, unsqueeze_default_33);  mul_tensor_22 = unsqueeze_default_33 = None
        unsqueeze_default_34: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_553, -1);  primals_553 = None
        unsqueeze_default_35: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, -1);  unsqueeze_default_34 = None
        add_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_23, unsqueeze_default_35);  mul_tensor_23 = unsqueeze_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_2: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        le_scalar_2: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_default_2, 0);  relu_default_2 = None
        where_self_2: "f32[128, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar_2, full_default_1, slice_tensor_4);  le_scalar_2 = slice_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_4: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_191, [0, 2, 3]);  getitem_191 = None
        unsqueeze_default_36: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_4, 0);  squeeze_dims_4 = None
        unsqueeze_default_37: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, 2);  unsqueeze_default_36 = None
        unsqueeze_default_38: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_37, 3);  unsqueeze_default_37 = None
        sum_dim_int_list_4: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self_2, [0, 2, 3])
        sub_tensor_9: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_default_38);  convolution_91 = unsqueeze_default_38 = None
        mul_tensor_24: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_2, sub_tensor_9)
        sum_dim_int_list_5: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 2, 3]);  mul_tensor_24 = None
        mul_tensor_25: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_4, 0.0001220703125);  sum_dim_int_list_4 = None
        unsqueeze_default_39: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_25, 0);  mul_tensor_25 = None
        unsqueeze_default_40: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 2);  unsqueeze_default_39 = None
        unsqueeze_default_41: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 3);  unsqueeze_default_40 = None
        mul_tensor_26: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, 0.0001220703125)
        squeeze_dims_5: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
        mul_tensor_27: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, squeeze_dims_5)
        mul_tensor_28: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_26, mul_tensor_27);  mul_tensor_26 = mul_tensor_27 = None
        unsqueeze_default_42: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_28, 0);  mul_tensor_28 = None
        unsqueeze_default_43: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 2);  unsqueeze_default_42 = None
        unsqueeze_default_44: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_43, 3);  unsqueeze_default_43 = None
        mul_tensor_29: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, primals_552);  primals_552 = None
        unsqueeze_default_45: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_29, 0);  mul_tensor_29 = None
        unsqueeze_default_46: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_45, 2);  unsqueeze_default_45 = None
        unsqueeze_default_47: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 3);  unsqueeze_default_46 = None
        mul_tensor_30: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_9, unsqueeze_default_44);  sub_tensor_9 = unsqueeze_default_44 = None
        sub_tensor_10: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_2, mul_tensor_30);  where_self_2 = mul_tensor_30 = None
        sub_tensor_11: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_10, unsqueeze_default_41);  sub_tensor_10 = unsqueeze_default_41 = None
        mul_tensor_31: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_11, unsqueeze_default_47);  sub_tensor_11 = unsqueeze_default_47 = None
        mul_tensor_32: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, squeeze_dims_5);  sum_dim_int_list_5 = squeeze_dims_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:226 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        slice_tensor_6: "f32[128, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_tensor_1, 1, 0, 384)
        slice_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_tensor_1, 1, 384, 768);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor_12: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_185)
        mul_tensor_33: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_12, rsqrt_88);  sub_tensor_12 = None
        unsqueeze_default_48: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_534, -1)
        unsqueeze_default_49: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, -1);  unsqueeze_default_48 = None
        mul_tensor_34: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_33, unsqueeze_default_49);  mul_tensor_33 = unsqueeze_default_49 = None
        unsqueeze_default_50: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_535, -1);  primals_535 = None
        unsqueeze_default_51: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_50, -1);  unsqueeze_default_50 = None
        add_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_34, unsqueeze_default_51);  mul_tensor_34 = unsqueeze_default_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_3: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        le_scalar_3: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_default_3, 0);  relu_default_3 = None
        where_self_3: "f32[128, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar_3, full_default_1, slice_tensor_7);  le_scalar_3 = slice_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_6: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3]);  getitem_185 = None
        unsqueeze_default_52: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_6, 0);  squeeze_dims_6 = None
        unsqueeze_default_53: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_52, 2);  unsqueeze_default_52 = None
        unsqueeze_default_54: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_53, 3);  unsqueeze_default_53 = None
        sum_dim_int_list_6: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self_3, [0, 2, 3])
        sub_tensor_13: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_default_54);  convolution_88 = unsqueeze_default_54 = None
        mul_tensor_35: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_3, sub_tensor_13)
        sum_dim_int_list_7: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 2, 3]);  mul_tensor_35 = None
        mul_tensor_36: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_6, 0.0001220703125);  sum_dim_int_list_6 = None
        unsqueeze_default_55: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_36, 0);  mul_tensor_36 = None
        unsqueeze_default_56: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_55, 2);  unsqueeze_default_55 = None
        unsqueeze_default_57: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_56, 3);  unsqueeze_default_56 = None
        mul_tensor_37: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_7, 0.0001220703125)
        squeeze_dims_7: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
        mul_tensor_38: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, squeeze_dims_7)
        mul_tensor_39: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_37, mul_tensor_38);  mul_tensor_37 = mul_tensor_38 = None
        unsqueeze_default_58: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_39, 0);  mul_tensor_39 = None
        unsqueeze_default_59: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_58, 2);  unsqueeze_default_58 = None
        unsqueeze_default_60: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_59, 3);  unsqueeze_default_59 = None
        mul_tensor_40: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, primals_534);  primals_534 = None
        unsqueeze_default_61: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_40, 0);  mul_tensor_40 = None
        unsqueeze_default_62: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_61, 2);  unsqueeze_default_61 = None
        unsqueeze_default_63: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_62, 3);  unsqueeze_default_62 = None
        mul_tensor_41: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_13, unsqueeze_default_60);  sub_tensor_13 = unsqueeze_default_60 = None
        sub_tensor_14: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_3, mul_tensor_41);  where_self_3 = mul_tensor_41 = None
        sub_tensor_15: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_14, unsqueeze_default_57);  sub_tensor_14 = unsqueeze_default_57 = None
        mul_tensor_42: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_15, unsqueeze_default_63);  sub_tensor_15 = unsqueeze_default_63 = None
        mul_tensor_43: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_7, squeeze_dims_7);  sum_dim_int_list_7 = squeeze_dims_7 = None
        sub_tensor_16: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_183)
        mul_tensor_44: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_16, rsqrt_87);  sub_tensor_16 = None
        unsqueeze_default_64: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_528, -1)
        unsqueeze_default_65: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_64, -1);  unsqueeze_default_64 = None
        mul_tensor_45: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_44, unsqueeze_default_65);  mul_tensor_44 = unsqueeze_default_65 = None
        unsqueeze_default_66: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_529, -1);  primals_529 = None
        unsqueeze_default_67: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_66, -1);  unsqueeze_default_66 = None
        add_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_45, unsqueeze_default_67);  mul_tensor_45 = unsqueeze_default_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_4: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        le_scalar_4: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_default_4, 0);  relu_default_4 = None
        where_self_4: "f32[128, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar_4, full_default_1, slice_tensor_6);  le_scalar_4 = slice_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_8: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
        unsqueeze_default_68: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_8, 0);  squeeze_dims_8 = None
        unsqueeze_default_69: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_68, 2);  unsqueeze_default_68 = None
        unsqueeze_default_70: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_69, 3);  unsqueeze_default_69 = None
        sum_dim_int_list_8: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self_4, [0, 2, 3])
        sub_tensor_17: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_default_70);  convolution_87 = unsqueeze_default_70 = None
        mul_tensor_46: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_4, sub_tensor_17)
        sum_dim_int_list_9: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [0, 2, 3]);  mul_tensor_46 = None
        mul_tensor_47: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_8, 0.0001220703125);  sum_dim_int_list_8 = None
        unsqueeze_default_71: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_47, 0);  mul_tensor_47 = None
        unsqueeze_default_72: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_71, 2);  unsqueeze_default_71 = None
        unsqueeze_default_73: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_72, 3);  unsqueeze_default_72 = None
        mul_tensor_48: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_9, 0.0001220703125)
        squeeze_dims_9: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
        mul_tensor_49: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_9, squeeze_dims_9)
        mul_tensor_50: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None
        unsqueeze_default_74: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_50, 0);  mul_tensor_50 = None
        unsqueeze_default_75: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_74, 2);  unsqueeze_default_74 = None
        unsqueeze_default_76: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_75, 3);  unsqueeze_default_75 = None
        mul_tensor_51: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_9, primals_528);  primals_528 = None
        unsqueeze_default_77: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_51, 0);  mul_tensor_51 = None
        unsqueeze_default_78: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_77, 2);  unsqueeze_default_77 = None
        unsqueeze_default_79: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_78, 3);  unsqueeze_default_78 = None
        mul_tensor_52: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_17, unsqueeze_default_76);  sub_tensor_17 = unsqueeze_default_76 = None
        sub_tensor_18: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_4, mul_tensor_52);  where_self_4 = mul_tensor_52 = None
        sub_tensor_19: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_18, unsqueeze_default_73);  sub_tensor_18 = unsqueeze_default_73 = None
        mul_tensor_53: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_19, unsqueeze_default_79);  sub_tensor_19 = unsqueeze_default_79 = None
        mul_tensor_54: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_9, squeeze_dims_9);  sum_dim_int_list_9 = squeeze_dims_9 = None
        sub_tensor_20: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_179)
        mul_tensor_55: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_20, rsqrt_85);  sub_tensor_20 = None
        unsqueeze_default_80: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_516, -1)
        unsqueeze_default_81: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_80, -1);  unsqueeze_default_80 = None
        mul_tensor_56: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_55, unsqueeze_default_81);  mul_tensor_55 = unsqueeze_default_81 = None
        unsqueeze_default_82: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_517, -1);  primals_517 = None
        unsqueeze_default_83: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_82, -1);  unsqueeze_default_82 = None
        add_tensor_5: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_56, unsqueeze_default_83);  mul_tensor_56 = unsqueeze_default_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_5: "f32[128, 320, 8, 8]" = torch.ops.aten.relu.default(add_tensor_5);  add_tensor_5 = None
        le_scalar_5: "b8[128, 320, 8, 8]" = torch.ops.aten.le.Scalar(relu_default_5, 0);  relu_default_5 = None
        where_self_5: "f32[128, 320, 8, 8]" = torch.ops.aten.where.self(le_scalar_5, full_default_1, slice_tensor);  le_scalar_5 = full_default_1 = slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_10: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
        unsqueeze_default_84: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_dims_10, 0);  squeeze_dims_10 = None
        unsqueeze_default_85: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_84, 2);  unsqueeze_default_84 = None
        unsqueeze_default_86: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_85, 3);  unsqueeze_default_85 = None
        sum_dim_int_list_10: "f32[320]" = torch.ops.aten.sum.dim_IntList(where_self_5, [0, 2, 3])
        sub_tensor_21: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_default_86);  convolution_85 = unsqueeze_default_86 = None
        mul_tensor_57: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_5, sub_tensor_21)
        sum_dim_int_list_11: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor_57, [0, 2, 3]);  mul_tensor_57 = None
        mul_tensor_58: "f32[320]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_10, 0.0001220703125);  sum_dim_int_list_10 = None
        unsqueeze_default_87: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_tensor_58, 0);  mul_tensor_58 = None
        unsqueeze_default_88: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_87, 2);  unsqueeze_default_87 = None
        unsqueeze_default_89: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_88, 3);  unsqueeze_default_88 = None
        mul_tensor_59: "f32[320]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_11, 0.0001220703125)
        squeeze_dims_11: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
        mul_tensor_60: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, squeeze_dims_11)
        mul_tensor_61: "f32[320]" = torch.ops.aten.mul.Tensor(mul_tensor_59, mul_tensor_60);  mul_tensor_59 = mul_tensor_60 = None
        unsqueeze_default_90: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_tensor_61, 0);  mul_tensor_61 = None
        unsqueeze_default_91: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_90, 2);  unsqueeze_default_90 = None
        unsqueeze_default_92: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_91, 3);  unsqueeze_default_91 = None
        mul_tensor_62: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, primals_516);  primals_516 = None
        unsqueeze_default_93: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_tensor_62, 0);  mul_tensor_62 = None
        unsqueeze_default_94: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_93, 2);  unsqueeze_default_93 = None
        unsqueeze_default_95: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_94, 3);  unsqueeze_default_94 = None
        mul_tensor_63: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_21, unsqueeze_default_92);  sub_tensor_21 = unsqueeze_default_92 = None
        sub_tensor_22: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_5, mul_tensor_63);  where_self_5 = mul_tensor_63 = None
        sub_tensor_23: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_22, unsqueeze_default_89);  sub_tensor_22 = unsqueeze_default_89 = None
        mul_tensor_64: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_23, unsqueeze_default_95);  sub_tensor_23 = unsqueeze_default_95 = None
        mul_tensor_65: "f32[320]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_11, squeeze_dims_11);  sum_dim_int_list_11 = squeeze_dims_11 = None
        return (mul_tensor_9, mul_tensor_10, mul_tensor_20, mul_tensor_21, mul_tensor_31, mul_tensor_32, mul_tensor_42, mul_tensor_43, mul_tensor_53, mul_tensor_54, mul_tensor_64, mul_tensor_65)

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
