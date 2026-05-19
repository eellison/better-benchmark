"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 65d7358072f1
Shape hash: 98c7ab2a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_474: "f32[128, 2048, 8, 8]", convolution_84: "f32[128, 192, 8, 8]", getitem_177: "f32[1, 192, 1, 1]", rsqrt_84: "f32[1, 192, 1, 1]", primals_510: "f32[192]", primals_511: "f32[192]", full_default: "f32[]", relu_80: "f32[128, 448, 8, 8]", getitem_232: "f32[128, 448, 8, 8]", convolution_80: "f32[128, 448, 8, 8]", unsqueeze_534: "f32[1, 448, 1, 1]", squeeze_241: "f32[448]", primals_486: "f32[448]", getitem_238: "f32[128, 384, 8, 8]", getitem_241: "f32[128, 384, 8, 8]", relu_77: "f32[128, 384, 8, 8]", convolution_77: "f32[128, 384, 8, 8]", unsqueeze_570: "f32[1, 384, 1, 1]", squeeze_232: "f32[384]", primals_468: "f32[384]", convolution_76: "f32[128, 320, 8, 8]", getitem_161: "f32[1, 320, 1, 1]", rsqrt_76: "f32[1, 320, 1, 1]", primals_462: "f32[320]", primals_463: "f32[320]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:244 in forward, code: return torch.cat(outputs, 1)
        slice_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.slice.Tensor(add_474, 1, 0, 320)
        slice_tensor_1: "f32[128, 192, 8, 8]" = torch.ops.aten.slice.Tensor(add_474, 1, 1856, 2048);  add_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_177)
        mul_tensor: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_84);  sub_tensor = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_510, -1)
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_511, -1);  primals_511 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 192, 8, 8]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[128, 192, 8, 8]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[128, 192, 8, 8]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor_1);  le_scalar = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3]);  getitem_177 = None
        unsqueeze_default_4: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_default_6);  convolution_84 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0001220703125);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0001220703125);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_510);  squeeze_dims_1 = primals_510 = None
        unsqueeze_default_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar_1: "b8[128, 448, 8, 8]" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None
        where_self_1: "f32[128, 448, 8, 8]" = torch.ops.aten.where.self(le_scalar_1, full_default, getitem_232);  le_scalar_1 = getitem_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list_2: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_4: "f32[128, 448, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_534);  convolution_80 = unsqueeze_534 = None
        mul_tensor_10: "f32[128, 448, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_4)
        sum_dim_int_list_3: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 2, 3]);  mul_tensor_10 = None
        mul_tensor_11: "f32[448]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.0001220703125);  sum_dim_int_list_2 = None
        unsqueeze_default_16: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_17: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_12: "f32[448]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.0001220703125);  sum_dim_int_list_3 = None
        mul_tensor_13: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
        mul_tensor_14: "f32[448]" = torch.ops.aten.mul.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        unsqueeze_default_19: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_20: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_15: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_486);  squeeze_241 = primals_486 = None
        unsqueeze_default_22: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_23: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_16: "f32[128, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_21);  sub_tensor_4 = unsqueeze_default_21 = None
        sub_tensor_5: "f32[128, 448, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_16);  where_self_1 = mul_tensor_16 = None
        sub_tensor_6: "f32[128, 448, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_5, unsqueeze_default_18);  sub_tensor_5 = unsqueeze_default_18 = None
        mul_tensor_17: "f32[128, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_24);  sub_tensor_6 = unsqueeze_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        add_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(getitem_238, getitem_241);  getitem_238 = getitem_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar_2: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_77, 0);  relu_77 = None
        where_self_2: "f32[128, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar_2, full_default, add_tensor_1);  le_scalar_2 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list_4: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self_2, [0, 2, 3])
        sub_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_570);  convolution_77 = unsqueeze_570 = None
        mul_tensor_18: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_2, sub_tensor_7)
        sum_dim_int_list_5: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 2, 3]);  mul_tensor_18 = None
        mul_tensor_19: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_4, 0.0001220703125);  sum_dim_int_list_4 = None
        unsqueeze_default_25: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_19, 0);  mul_tensor_19 = None
        unsqueeze_default_26: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 2);  unsqueeze_default_25 = None
        unsqueeze_default_27: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 3);  unsqueeze_default_26 = None
        mul_tensor_20: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, 0.0001220703125);  sum_dim_int_list_5 = None
        mul_tensor_21: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
        mul_tensor_22: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        unsqueeze_default_28: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_22, 0);  mul_tensor_22 = None
        unsqueeze_default_29: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 2);  unsqueeze_default_28 = None
        unsqueeze_default_30: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 3);  unsqueeze_default_29 = None
        mul_tensor_23: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_468);  squeeze_232 = primals_468 = None
        unsqueeze_default_31: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_23, 0);  mul_tensor_23 = None
        unsqueeze_default_32: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_31, 2);  unsqueeze_default_31 = None
        unsqueeze_default_33: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 3);  unsqueeze_default_32 = None
        mul_tensor_24: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_30);  sub_tensor_7 = unsqueeze_default_30 = None
        sub_tensor_8: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_2, mul_tensor_24);  where_self_2 = mul_tensor_24 = None
        sub_tensor_9: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_8, unsqueeze_default_27);  sub_tensor_8 = unsqueeze_default_27 = None
        mul_tensor_25: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_9, unsqueeze_default_33);  sub_tensor_9 = unsqueeze_default_33 = None
        sub_tensor_10: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_161)
        mul_tensor_26: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_10, rsqrt_76);  sub_tensor_10 = None
        unsqueeze_default_34: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_462, -1)
        unsqueeze_default_35: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, -1);  unsqueeze_default_34 = None
        mul_tensor_27: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_26, unsqueeze_default_35);  mul_tensor_26 = unsqueeze_default_35 = None
        unsqueeze_default_36: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_463, -1);  primals_463 = None
        unsqueeze_default_37: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, -1);  unsqueeze_default_36 = None
        add_tensor_2: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_27, unsqueeze_default_37);  mul_tensor_27 = unsqueeze_default_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 320, 8, 8]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        le_scalar_3: "b8[128, 320, 8, 8]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        where_self_3: "f32[128, 320, 8, 8]" = torch.ops.aten.where.self(le_scalar_3, full_default, slice_tensor);  le_scalar_3 = full_default = slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_2: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
        unsqueeze_default_38: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_39: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 2);  unsqueeze_default_38 = None
        unsqueeze_default_40: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 3);  unsqueeze_default_39 = None
        sum_dim_int_list_6: "f32[320]" = torch.ops.aten.sum.dim_IntList(where_self_3, [0, 2, 3])
        sub_tensor_11: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_default_40);  convolution_76 = unsqueeze_default_40 = None
        mul_tensor_28: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(where_self_3, sub_tensor_11)
        sum_dim_int_list_7: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [0, 2, 3]);  mul_tensor_28 = None
        mul_tensor_29: "f32[320]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_6, 0.0001220703125);  sum_dim_int_list_6 = None
        unsqueeze_default_41: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_tensor_29, 0);  mul_tensor_29 = None
        unsqueeze_default_42: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_41, 2);  unsqueeze_default_41 = None
        unsqueeze_default_43: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 3);  unsqueeze_default_42 = None
        mul_tensor_30: "f32[320]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_7, 0.0001220703125);  sum_dim_int_list_7 = None
        squeeze_dims_3: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
        mul_tensor_31: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_32: "f32[320]" = torch.ops.aten.mul.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        unsqueeze_default_44: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_tensor_32, 0);  mul_tensor_32 = None
        unsqueeze_default_45: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, 2);  unsqueeze_default_44 = None
        unsqueeze_default_46: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_45, 3);  unsqueeze_default_45 = None
        mul_tensor_33: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, primals_462);  squeeze_dims_3 = primals_462 = None
        unsqueeze_default_47: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_tensor_33, 0);  mul_tensor_33 = None
        unsqueeze_default_48: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_47, 2);  unsqueeze_default_47 = None
        unsqueeze_default_49: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, 3);  unsqueeze_default_48 = None
        mul_tensor_34: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_11, unsqueeze_default_46);  sub_tensor_11 = unsqueeze_default_46 = None
        sub_tensor_12: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(where_self_3, mul_tensor_34);  where_self_3 = mul_tensor_34 = None
        sub_tensor_13: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_12, unsqueeze_default_43);  sub_tensor_12 = unsqueeze_default_43 = None
        mul_tensor_35: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_13, unsqueeze_default_49);  sub_tensor_13 = unsqueeze_default_49 = None
        return (mul_tensor_9, mul_tensor_17, mul_tensor_25, mul_tensor_35)


def _default_make_inputs():
    return [
    torch.randn(16777216, dtype=torch.float32, device='cuda').as_strided([128, 2048, 8, 8], [131072, 1, 16384, 2048]),  # add_474
    torch.randn(1572864, dtype=torch.float32, device='cuda').as_strided([128, 192, 8, 8], [12288, 1, 1536, 192]),  # convolution_84
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(3670016, dtype=torch.float32, device='cuda').as_strided([128, 448, 8, 8], [28672, 1, 3584, 448]),  # relu_80
    torch.randn(3670016, dtype=torch.float32, device='cuda').as_strided([128, 448, 8, 8], [28672, 1, 3584, 448]),  # getitem_232
    torch.randn(3670016, dtype=torch.float32, device='cuda').as_strided([128, 448, 8, 8], [28672, 1, 3584, 448]),  # convolution_80
    torch.randn([1, 448, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([128, 384, 8, 8], [24576, 1, 3072, 384]),  # getitem_238
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([128, 384, 8, 8], [24576, 1, 3072, 384]),  # getitem_241
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([128, 384, 8, 8], [24576, 1, 3072, 384]),  # relu_77
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([128, 384, 8, 8], [24576, 1, 3072, 384]),  # convolution_77
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(2621440, dtype=torch.float32, device='cuda').as_strided([128, 320, 8, 8], [20480, 1, 2560, 320]),  # convolution_76
    torch.randn([1, 320, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 320, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
