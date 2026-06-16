"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 8d5cb645d3bf
Shape hash: 13e49e96
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 320, 8, 8]", arg1_1: "f32[320]", arg2_1: "f32[320]", arg3_1: "f32[320]", arg4_1: "f32[320]", arg5_1: "bf16[128, 384, 8, 8]", arg6_1: "f32[384]", arg7_1: "f32[384]", arg8_1: "f32[384]", arg9_1: "f32[384]", arg10_1: "bf16[128, 384, 8, 8]", arg11_1: "f32[384]", arg12_1: "f32[384]", arg13_1: "f32[384]", arg14_1: "f32[384]", arg15_1: "bf16[128, 384, 8, 8]", arg16_1: "f32[384]", arg17_1: "f32[384]", arg18_1: "f32[384]", arg19_1: "f32[384]", arg20_1: "bf16[128, 384, 8, 8]", arg21_1: "f32[384]", arg22_1: "f32[384]", arg23_1: "f32[384]", arg24_1: "f32[384]", arg25_1: "bf16[128, 192, 8, 8]", arg26_1: "f32[192]", arg27_1: "f32[192]", arg28_1: "f32[192]", arg29_1: "f32[192]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 320, 8, 8]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 320, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 320, 1, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = None
        mul: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[320]" = torch.ops.aten.mul.Tensor(arg1_1, 0.9)
        add_1: "f32[320]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_1: "f32[320]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_1, 1.0001220852154804);  squeeze_1 = None
        mul_4: "f32[320]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[320]" = torch.ops.aten.mul.Tensor(arg2_1, 0.9)
        add_2: "f32[320]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_1: "bf16[128, 320, 8, 8]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        relu: "bf16[128, 320, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None
        convert_element_type_2: "f32[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem_2: "f32[1, 384, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 384, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
        rsqrt_1: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_1: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(arg5_1, getitem_3);  arg5_1 = None
        mul_7: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_2: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3])
        mul_8: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_2, 0.1);  squeeze_2 = None
        mul_9: "f32[384]" = torch.ops.aten.mul.Tensor(arg6_1, 0.9)
        add_5: "f32[384]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_3: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_3, 1.0001220852154804);  squeeze_3 = None
        mul_11: "f32[384]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[384]" = torch.ops.aten.mul.Tensor(arg7_1, 0.9)
        add_6: "f32[384]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_5: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_7: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_7: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_3: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        relu_1: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None
        convert_element_type_4: "f32[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_4, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_4 = None
        getitem_4: "f32[1, 384, 1, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 384, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_8: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
        rsqrt_2: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(arg10_1, getitem_5);  arg10_1 = None
        mul_14: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_4: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3])
        mul_15: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_4, 0.1);  squeeze_4 = None
        mul_16: "f32[384]" = torch.ops.aten.mul.Tensor(arg11_1, 0.9)
        add_9: "f32[384]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_5: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_17: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0001220852154804);  squeeze_5 = None
        mul_18: "f32[384]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[384]" = torch.ops.aten.mul.Tensor(arg12_1, 0.9)
        add_10: "f32[384]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_9: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_11: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_11: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convert_element_type_5: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        relu_2: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_5);  convert_element_type_5 = None
        cat: "bf16[128, 768, 8, 8]" = torch.ops.aten.cat.default([relu_1, relu_2], 1);  relu_1 = relu_2 = None
        convert_element_type_6: "f32[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_6, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_6 = None
        getitem_6: "f32[1, 384, 1, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 384, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_12: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 0.001)
        rsqrt_3: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_3: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(arg15_1, getitem_7);  arg15_1 = None
        mul_21: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_6: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3])
        mul_22: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_23: "f32[384]" = torch.ops.aten.mul.Tensor(arg16_1, 0.9)
        add_13: "f32[384]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_7: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_24: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_7, 1.0001220852154804);  squeeze_7 = None
        mul_25: "f32[384]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[384]" = torch.ops.aten.mul.Tensor(arg17_1, 0.9)
        add_14: "f32[384]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_13: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_15: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_15: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        convert_element_type_7: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None
        relu_3: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_7);  convert_element_type_7 = None
        convert_element_type_8: "f32[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_8, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_8 = None
        getitem_8: "f32[1, 384, 1, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 384, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_16: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 0.001)
        rsqrt_4: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_4: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(arg20_1, getitem_9);  arg20_1 = None
        mul_28: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_8: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3])
        mul_29: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_8, 0.1);  squeeze_8 = None
        mul_30: "f32[384]" = torch.ops.aten.mul.Tensor(arg21_1, 0.9)
        add_17: "f32[384]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_9: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_31: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_9, 1.0001220852154804);  squeeze_9 = None
        mul_32: "f32[384]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[384]" = torch.ops.aten.mul.Tensor(arg22_1, 0.9)
        add_18: "f32[384]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_17: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_19: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_19: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        convert_element_type_9: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        relu_4: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_9);  convert_element_type_9 = None
        cat_1: "bf16[128, 768, 8, 8]" = torch.ops.aten.cat.default([relu_3, relu_4], 1);  relu_3 = relu_4 = None
        convert_element_type_10: "f32[128, 192, 8, 8]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_10, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_10 = None
        getitem_10: "f32[1, 192, 1, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 192, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_20: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 0.001)
        rsqrt_5: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_5: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(arg25_1, getitem_11);  arg25_1 = None
        mul_35: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_10: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3])
        mul_36: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_10, 0.1);  squeeze_10 = None
        mul_37: "f32[192]" = torch.ops.aten.mul.Tensor(arg26_1, 0.9)
        add_21: "f32[192]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_11: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_38: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0001220852154804);  squeeze_11 = None
        mul_39: "f32[192]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[192]" = torch.ops.aten.mul.Tensor(arg27_1, 0.9)
        add_22: "f32[192]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_21: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_23: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_23: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        convert_element_type_11: "bf16[128, 192, 8, 8]" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None
        relu_5: "bf16[128, 192, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_11);  convert_element_type_11 = None
        cat_2: "bf16[128, 2048, 8, 8]" = torch.ops.aten.cat.default([relu, cat, cat_1, relu_5], 1);  relu = cat = cat_1 = relu_5 = None
        mean: "bf16[128, 2048, 1, 1]" = torch.ops.aten.mean.dim(cat_2, [-1, -2], True);  cat_2 = None
        as_strided: "bf16[128, 2048, 1, 1]" = torch.ops.aten.as_strided.default(mean, _shape_param_0, _shape_param_1);  mean = _shape_param_0 = _shape_param_1 = None
        view: "bf16[128, 2048]" = torch.ops.aten.view.default(as_strided, _shape_param_2);  as_strided = _shape_param_2 = None
        copy_: "f32[320]" = torch.ops.aten.copy_.default(arg1_1, add_1);  arg1_1 = add_1 = None
        copy__1: "f32[320]" = torch.ops.aten.copy_.default(arg2_1, add_2);  arg2_1 = add_2 = None
        copy__2: "f32[384]" = torch.ops.aten.copy_.default(arg6_1, add_5);  arg6_1 = add_5 = None
        copy__3: "f32[384]" = torch.ops.aten.copy_.default(arg7_1, add_6);  arg7_1 = add_6 = None
        copy__4: "f32[384]" = torch.ops.aten.copy_.default(arg11_1, add_9);  arg11_1 = add_9 = None
        copy__5: "f32[384]" = torch.ops.aten.copy_.default(arg12_1, add_10);  arg12_1 = add_10 = None
        copy__6: "f32[384]" = torch.ops.aten.copy_.default(arg16_1, add_13);  arg16_1 = add_13 = None
        copy__7: "f32[384]" = torch.ops.aten.copy_.default(arg17_1, add_14);  arg17_1 = add_14 = None
        copy__8: "f32[384]" = torch.ops.aten.copy_.default(arg21_1, add_17);  arg21_1 = add_17 = None
        copy__9: "f32[384]" = torch.ops.aten.copy_.default(arg22_1, add_18);  arg22_1 = add_18 = None
        copy__10: "f32[192]" = torch.ops.aten.copy_.default(arg26_1, add_21);  arg26_1 = add_21 = None
        copy__11: "f32[192]" = torch.ops.aten.copy_.default(arg27_1, add_22);  arg27_1 = add_22 = None
        return (getitem_1, rsqrt, getitem_3, rsqrt_1, getitem_5, rsqrt_2, getitem_7, rsqrt_3, getitem_9, rsqrt_4, getitem_11, rsqrt_5, view, copy_, copy__1, copy__2, copy__3, copy__4, copy__5, copy__6, copy__7, copy__8, copy__9, copy__10, copy__11)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
