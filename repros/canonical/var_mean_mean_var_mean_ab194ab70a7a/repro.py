"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: ab194ab70a7a
Shape hash: 3f5b3c4b
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
_shapes_config = "(T([128, 320, 8, 8], f32, stride=(20480, 1, 2560, 320)), T([320], f32), T([320], f32), T([320], f32), T([320], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 192, 8, 8], f32, stride=(12288, 1, 1536, 192)), T([192], f32), T([192], f32), T([192], f32), T([192], f32), S([128, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_85: "f32[128, 320, 8, 8]", primals_514: "f32[320]", primals_515: "f32[320]", primals_516: "f32[320]", primals_517: "f32[320]", convolution_87: "f32[128, 384, 8, 8]", primals_526: "f32[384]", primals_527: "f32[384]", primals_528: "f32[384]", primals_529: "f32[384]", convolution_88: "f32[128, 384, 8, 8]", primals_532: "f32[384]", primals_533: "f32[384]", primals_534: "f32[384]", primals_535: "f32[384]", convolution_91: "f32[128, 384, 8, 8]", primals_550: "f32[384]", primals_551: "f32[384]", primals_552: "f32[384]", primals_553: "f32[384]", convolution_92: "f32[128, 384, 8, 8]", primals_556: "f32[384]", primals_557: "f32[384]", primals_558: "f32[384]", primals_559: "f32[384]", convolution_93: "f32[128, 192, 8, 8]", primals_562: "f32[192]", primals_563: "f32[192]", primals_564: "f32[192]", primals_565: "f32[192]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 320, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 320, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt_default: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_1);  convolution_85 = None
        mul_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[320]" = torch.ops.aten.mul.Tensor(primals_514, 0.9)
        add_tensor_1: "f32[320]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[320]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0001220852154804);  squeeze_dims_1 = None
        mul_tensor_4: "f32[320]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[320]" = torch.ops.aten.mul.Tensor(primals_515, 0.9)
        add_tensor_2: "f32[320]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_516, -1);  primals_516 = None
        unsqueeze_default_1: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_517, -1);  primals_517 = None
        unsqueeze_default_3: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 320, 8, 8]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_87, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 384, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 384, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
        rsqrt_default_1: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_3);  convolution_87 = None
        mul_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        squeeze_dims_2: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        mul_tensor_8: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 0.1);  squeeze_dims_2 = None
        mul_tensor_9: "f32[384]" = torch.ops.aten.mul.Tensor(primals_526, 0.9)
        add_tensor_5: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        squeeze_dims_3: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_10: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 1.0001220852154804);  squeeze_dims_3 = None
        mul_tensor_11: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.1);  mul_tensor_10 = None
        mul_tensor_12: "f32[384]" = torch.ops.aten.mul.Tensor(primals_527, 0.9)
        add_tensor_6: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_4: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_528, -1);  primals_528 = None
        unsqueeze_default_5: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_13: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_5);  mul_tensor_7 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_529, -1);  primals_529 = None
        unsqueeze_default_7: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_13, unsqueeze_default_7);  mul_tensor_13 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 384, 1, 1]" = var_mean_correction_2[0]
        getitem_5: "f32[1, 384, 1, 1]" = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_8: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
        rsqrt_default_2: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_8);  add_tensor_8 = None
        sub_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_5);  convolution_88 = None
        mul_tensor_14: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = rsqrt_default_2 = None
        squeeze_dims_4: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_4, 0.1);  squeeze_dims_4 = None
        mul_tensor_16: "f32[384]" = torch.ops.aten.mul.Tensor(primals_532, 0.9)
        add_tensor_9: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_5: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_17: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.0001220852154804);  squeeze_dims_5 = None
        mul_tensor_18: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[384]" = torch.ops.aten.mul.Tensor(primals_533, 0.9)
        add_tensor_10: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
        unsqueeze_default_8: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_534, -1);  primals_534 = None
        unsqueeze_default_9: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        mul_tensor_20: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_14, unsqueeze_default_9);  mul_tensor_14 = unsqueeze_default_9 = None
        unsqueeze_default_10: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_535, -1);  primals_535 = None
        unsqueeze_default_11: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        add_tensor_11: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_20, unsqueeze_default_11);  mul_tensor_20 = unsqueeze_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_2: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_11);  add_tensor_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:226 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        cat_default: "f32[128, 768, 8, 8]" = torch.ops.aten.cat.default([relu_default_1, relu_default_2], 1);  relu_default_1 = relu_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convolution_91, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 384, 1, 1]" = var_mean_correction_3[0]
        getitem_7: "f32[1, 384, 1, 1]" = var_mean_correction_3[1];  var_mean_correction_3 = None
        add_tensor_12: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 0.001)
        rsqrt_default_3: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_12);  add_tensor_12 = None
        sub_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_91, getitem_7);  convolution_91 = None
        mul_tensor_21: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_3);  sub_tensor_3 = rsqrt_default_3 = None
        squeeze_dims_6: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        mul_tensor_22: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_6, 0.1);  squeeze_dims_6 = None
        mul_tensor_23: "f32[384]" = torch.ops.aten.mul.Tensor(primals_550, 0.9)
        add_tensor_13: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_22, mul_tensor_23);  mul_tensor_22 = mul_tensor_23 = None
        squeeze_dims_7: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_24: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, 1.0001220852154804);  squeeze_dims_7 = None
        mul_tensor_25: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_24, 0.1);  mul_tensor_24 = None
        mul_tensor_26: "f32[384]" = torch.ops.aten.mul.Tensor(primals_551, 0.9)
        add_tensor_14: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        unsqueeze_default_12: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_552, -1);  primals_552 = None
        unsqueeze_default_13: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_27: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_21, unsqueeze_default_13);  mul_tensor_21 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_553, -1);  primals_553 = None
        unsqueeze_default_15: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_15: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_27, unsqueeze_default_15);  mul_tensor_27 = unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_3: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_15);  add_tensor_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 384, 1, 1]" = var_mean_correction_4[0]
        getitem_9: "f32[1, 384, 1, 1]" = var_mean_correction_4[1];  var_mean_correction_4 = None
        add_tensor_16: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 0.001)
        rsqrt_default_4: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_16);  add_tensor_16 = None
        sub_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_9);  convolution_92 = None
        mul_tensor_28: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_default_4);  sub_tensor_4 = rsqrt_default_4 = None
        squeeze_dims_8: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        mul_tensor_29: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 0.1);  squeeze_dims_8 = None
        mul_tensor_30: "f32[384]" = torch.ops.aten.mul.Tensor(primals_556, 0.9)
        add_tensor_17: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_29, mul_tensor_30);  mul_tensor_29 = mul_tensor_30 = None
        squeeze_dims_9: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_tensor_31: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_9, 1.0001220852154804);  squeeze_dims_9 = None
        mul_tensor_32: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_31, 0.1);  mul_tensor_31 = None
        mul_tensor_33: "f32[384]" = torch.ops.aten.mul.Tensor(primals_557, 0.9)
        add_tensor_18: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_32, mul_tensor_33);  mul_tensor_32 = mul_tensor_33 = None
        unsqueeze_default_16: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_558, -1);  primals_558 = None
        unsqueeze_default_17: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        mul_tensor_34: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_28, unsqueeze_default_17);  mul_tensor_28 = unsqueeze_default_17 = None
        unsqueeze_default_18: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_559, -1);  primals_559 = None
        unsqueeze_default_19: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        add_tensor_19: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_34, unsqueeze_default_19);  mul_tensor_34 = unsqueeze_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_4: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_19);  add_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:234 in _forward, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
        cat_default_1: "f32[128, 768, 8, 8]" = torch.ops.aten.cat.default([relu_default_3, relu_default_4], 1);  relu_default_3 = relu_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[1, 192, 1, 1]" = var_mean_correction_5[0]
        getitem_11: "f32[1, 192, 1, 1]" = var_mean_correction_5[1];  var_mean_correction_5 = None
        add_tensor_20: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 0.001)
        rsqrt_default_5: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_20);  add_tensor_20 = None
        sub_tensor_5: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_11);  convolution_93 = None
        mul_tensor_35: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_5, rsqrt_default_5);  sub_tensor_5 = rsqrt_default_5 = None
        squeeze_dims_10: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        mul_tensor_36: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_10, 0.1);  squeeze_dims_10 = None
        mul_tensor_37: "f32[192]" = torch.ops.aten.mul.Tensor(primals_562, 0.9)
        add_tensor_21: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_36, mul_tensor_37);  mul_tensor_36 = mul_tensor_37 = None
        squeeze_dims_11: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_tensor_38: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, 1.0001220852154804);  squeeze_dims_11 = None
        mul_tensor_39: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_38, 0.1);  mul_tensor_38 = None
        mul_tensor_40: "f32[192]" = torch.ops.aten.mul.Tensor(primals_563, 0.9)
        add_tensor_22: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_39, mul_tensor_40);  mul_tensor_39 = mul_tensor_40 = None
        unsqueeze_default_20: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_564, -1);  primals_564 = None
        unsqueeze_default_21: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, -1);  unsqueeze_default_20 = None
        mul_tensor_41: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_35, unsqueeze_default_21);  mul_tensor_35 = unsqueeze_default_21 = None
        unsqueeze_default_22: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_565, -1);  primals_565 = None
        unsqueeze_default_23: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, -1);  unsqueeze_default_22 = None
        add_tensor_23: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_41, unsqueeze_default_23);  mul_tensor_41 = unsqueeze_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_5: "f32[128, 192, 8, 8]" = torch.ops.aten.relu.default(add_tensor_23);  add_tensor_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:244 in forward, code: return torch.cat(outputs, 1)
        cat_default_2: "f32[128, 2048, 8, 8]" = torch.ops.aten.cat.default([relu_default, cat_default, cat_default_1, relu_default_5], 1);  relu_default = cat_default = cat_default_1 = relu_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[128, 2048, 1, 1]" = torch.ops.aten.mean.dim(cat_default_2, [-1, -2], True);  cat_default_2 = None
        as_strided_default: "f32[128, 2048, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 2048, 1, 1], [2048, 1, 2048, 2048]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 2048]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None

        # No stacktrace found for following nodes
        copy__default: "f32[320]" = torch.ops.aten.copy_.default(primals_514, add_tensor_1);  primals_514 = add_tensor_1 = None
        copy__default_1: "f32[320]" = torch.ops.aten.copy_.default(primals_515, add_tensor_2);  primals_515 = add_tensor_2 = None
        copy__default_2: "f32[384]" = torch.ops.aten.copy_.default(primals_526, add_tensor_5);  primals_526 = add_tensor_5 = None
        copy__default_3: "f32[384]" = torch.ops.aten.copy_.default(primals_527, add_tensor_6);  primals_527 = add_tensor_6 = None
        copy__default_4: "f32[384]" = torch.ops.aten.copy_.default(primals_532, add_tensor_9);  primals_532 = add_tensor_9 = None
        copy__default_5: "f32[384]" = torch.ops.aten.copy_.default(primals_533, add_tensor_10);  primals_533 = add_tensor_10 = None
        copy__default_6: "f32[384]" = torch.ops.aten.copy_.default(primals_550, add_tensor_13);  primals_550 = add_tensor_13 = None
        copy__default_7: "f32[384]" = torch.ops.aten.copy_.default(primals_551, add_tensor_14);  primals_551 = add_tensor_14 = None
        copy__default_8: "f32[384]" = torch.ops.aten.copy_.default(primals_556, add_tensor_17);  primals_556 = add_tensor_17 = None
        copy__default_9: "f32[384]" = torch.ops.aten.copy_.default(primals_557, add_tensor_18);  primals_557 = add_tensor_18 = None
        copy__default_10: "f32[192]" = torch.ops.aten.copy_.default(primals_562, add_tensor_21);  primals_562 = add_tensor_21 = None
        copy__default_11: "f32[192]" = torch.ops.aten.copy_.default(primals_563, add_tensor_22);  primals_563 = add_tensor_22 = None
        return (copy__default, copy__default_1, reshape_default, copy__default_10, copy__default_8, copy__default_6, copy__default_2, copy__default_4, copy__default_11, copy__default_9, copy__default_7, copy__default_3, copy__default_5)



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
