"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s1_g10
Pattern hash: 2949fb044264
Shape hash: 7c816b45
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg427_1: "bf16[320]", arg428_1: "bf16[320]", convolution_85: "bf16[128, 320, 8, 8]", arg429_1: "bf16[320]", arg430_1: "bf16[320]", arg437_1: "bf16[384]", arg438_1: "bf16[384]", convolution_87: "bf16[128, 384, 8, 8]", arg439_1: "bf16[384]", arg440_1: "bf16[384]", arg442_1: "bf16[384]", arg443_1: "bf16[384]", convolution_88: "bf16[128, 384, 8, 8]", arg444_1: "bf16[384]", arg445_1: "bf16[384]", arg457_1: "bf16[384]", arg458_1: "bf16[384]", convolution_91: "bf16[128, 384, 8, 8]", arg459_1: "bf16[384]", arg460_1: "bf16[384]", arg462_1: "bf16[384]", arg463_1: "bf16[384]", convolution_92: "bf16[128, 384, 8, 8]", arg464_1: "bf16[384]", arg465_1: "bf16[384]", arg467_1: "bf16[192]", arg468_1: "bf16[192]", convolution_93: "bf16[128, 192, 8, 8]", arg469_1: "bf16[192]", arg470_1: "bf16[192]", arg471_1: "bf16[1000, 2048]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[320]" = torch.ops.prims.convert_element_type.default(arg427_1, torch.float32);  arg427_1 = None
        convert_element_type_default_1: "f32[320]" = torch.ops.prims.convert_element_type.default(arg428_1, torch.float32);  arg428_1 = None
        add_tensor: "f32[320]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 0.001);  convert_element_type_default_1 = None
        sqrt_default: "f32[320]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        sub_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_default_1);  convolution_85 = unsqueeze_default_1 = None
        mul_tensor_1: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "bf16[320, 1]" = torch.ops.aten.unsqueeze.default(arg429_1, -1);  arg429_1 = None
        unsqueeze_default_5: "bf16[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "bf16[320, 1]" = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
        unsqueeze_default_7: "bf16[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "bf16[128, 320, 8, 8]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        relu_default: "bf16[128, 320, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        convert_element_type_default_3: "f32[384]" = torch.ops.prims.convert_element_type.default(arg437_1, torch.float32);  arg437_1 = None
        convert_element_type_default_4: "f32[384]" = torch.ops.prims.convert_element_type.default(arg438_1, torch.float32);  arg438_1 = None
        add_tensor_2: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_default_4, 0.001);  convert_element_type_default_4 = None
        sqrt_default_1: "f32[384]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_8: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, -1);  convert_element_type_default_3 = None
        unsqueeze_default_9: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        sub_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_default_9);  convolution_87 = unsqueeze_default_9 = None
        mul_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(arg439_1, -1);  arg439_1 = None
        unsqueeze_default_13: "bf16[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(arg440_1, -1);  arg440_1 = None
        unsqueeze_default_15: "bf16[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        convert_element_type_default_5: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        relu_default_1: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_default_5);  convert_element_type_default_5 = None
        convert_element_type_default_6: "f32[384]" = torch.ops.prims.convert_element_type.default(arg442_1, torch.float32);  arg442_1 = None
        convert_element_type_default_7: "f32[384]" = torch.ops.prims.convert_element_type.default(arg443_1, torch.float32);  arg443_1 = None
        add_tensor_4: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_default_7, 0.001);  convert_element_type_default_7 = None
        sqrt_default_2: "f32[384]" = torch.ops.aten.sqrt.default(add_tensor_4);  add_tensor_4 = None
        reciprocal_default_2: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_default_2);  sqrt_default_2 = None
        mul_tensor_6: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_default_2, 1);  reciprocal_default_2 = None
        unsqueeze_default_16: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_6, -1);  convert_element_type_default_6 = None
        unsqueeze_default_17: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, -1);  mul_tensor_6 = None
        unsqueeze_default_19: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        sub_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_default_17);  convolution_88 = unsqueeze_default_17 = None
        mul_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_19);  sub_tensor_2 = unsqueeze_default_19 = None
        unsqueeze_default_20: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
        unsqueeze_default_21: "bf16[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, -1);  unsqueeze_default_20 = None
        mul_tensor_8: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_21);  mul_tensor_7 = unsqueeze_default_21 = None
        unsqueeze_default_22: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(arg445_1, -1);  arg445_1 = None
        unsqueeze_default_23: "bf16[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, -1);  unsqueeze_default_22 = None
        add_tensor_5: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_8, unsqueeze_default_23);  mul_tensor_8 = unsqueeze_default_23 = None
        convert_element_type_default_8: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16);  add_tensor_5 = None
        relu_default_2: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_default_8);  convert_element_type_default_8 = None
        cat_default: "bf16[128, 768, 8, 8]" = torch.ops.aten.cat.default([relu_default_1, relu_default_2], 1);  relu_default_1 = relu_default_2 = None
        convert_element_type_default_9: "f32[384]" = torch.ops.prims.convert_element_type.default(arg457_1, torch.float32);  arg457_1 = None
        convert_element_type_default_10: "f32[384]" = torch.ops.prims.convert_element_type.default(arg458_1, torch.float32);  arg458_1 = None
        add_tensor_6: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_default_10, 0.001);  convert_element_type_default_10 = None
        sqrt_default_3: "f32[384]" = torch.ops.aten.sqrt.default(add_tensor_6);  add_tensor_6 = None
        reciprocal_default_3: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_default_3);  sqrt_default_3 = None
        mul_tensor_9: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_default_3, 1);  reciprocal_default_3 = None
        unsqueeze_default_24: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_9, -1);  convert_element_type_default_9 = None
        unsqueeze_default_25: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, -1);  unsqueeze_default_24 = None
        unsqueeze_default_26: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, -1);  mul_tensor_9 = None
        unsqueeze_default_27: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, -1);  unsqueeze_default_26 = None
        sub_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_default_25);  convolution_91 = unsqueeze_default_25 = None
        mul_tensor_10: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_27);  sub_tensor_3 = unsqueeze_default_27 = None
        unsqueeze_default_28: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(arg459_1, -1);  arg459_1 = None
        unsqueeze_default_29: "bf16[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, -1);  unsqueeze_default_28 = None
        mul_tensor_11: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_10, unsqueeze_default_29);  mul_tensor_10 = unsqueeze_default_29 = None
        unsqueeze_default_30: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(arg460_1, -1);  arg460_1 = None
        unsqueeze_default_31: "bf16[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, -1);  unsqueeze_default_30 = None
        add_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_11, unsqueeze_default_31);  mul_tensor_11 = unsqueeze_default_31 = None
        convert_element_type_default_11: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.bfloat16);  add_tensor_7 = None
        relu_default_3: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_default_11);  convert_element_type_default_11 = None
        convert_element_type_default_12: "f32[384]" = torch.ops.prims.convert_element_type.default(arg462_1, torch.float32);  arg462_1 = None
        convert_element_type_default_13: "f32[384]" = torch.ops.prims.convert_element_type.default(arg463_1, torch.float32);  arg463_1 = None
        add_tensor_8: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_default_13, 0.001);  convert_element_type_default_13 = None
        sqrt_default_4: "f32[384]" = torch.ops.aten.sqrt.default(add_tensor_8);  add_tensor_8 = None
        reciprocal_default_4: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_default_4);  sqrt_default_4 = None
        mul_tensor_12: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_default_4, 1);  reciprocal_default_4 = None
        unsqueeze_default_32: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_12, -1);  convert_element_type_default_12 = None
        unsqueeze_default_33: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, -1);  unsqueeze_default_32 = None
        unsqueeze_default_34: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, -1);  mul_tensor_12 = None
        unsqueeze_default_35: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, -1);  unsqueeze_default_34 = None
        sub_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_default_33);  convolution_92 = unsqueeze_default_33 = None
        mul_tensor_13: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_35);  sub_tensor_4 = unsqueeze_default_35 = None
        unsqueeze_default_36: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_default_37: "bf16[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, -1);  unsqueeze_default_36 = None
        mul_tensor_14: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_13, unsqueeze_default_37);  mul_tensor_13 = unsqueeze_default_37 = None
        unsqueeze_default_38: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(arg465_1, -1);  arg465_1 = None
        unsqueeze_default_39: "bf16[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, -1);  unsqueeze_default_38 = None
        add_tensor_9: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_14, unsqueeze_default_39);  mul_tensor_14 = unsqueeze_default_39 = None
        convert_element_type_default_14: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_tensor_9, torch.bfloat16);  add_tensor_9 = None
        relu_default_4: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_default_14);  convert_element_type_default_14 = None
        cat_default_1: "bf16[128, 768, 8, 8]" = torch.ops.aten.cat.default([relu_default_3, relu_default_4], 1);  relu_default_3 = relu_default_4 = None
        convert_element_type_default_15: "f32[192]" = torch.ops.prims.convert_element_type.default(arg467_1, torch.float32);  arg467_1 = None
        convert_element_type_default_16: "f32[192]" = torch.ops.prims.convert_element_type.default(arg468_1, torch.float32);  arg468_1 = None
        add_tensor_10: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_default_16, 0.001);  convert_element_type_default_16 = None
        sqrt_default_5: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor_10);  add_tensor_10 = None
        reciprocal_default_5: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default_5);  sqrt_default_5 = None
        mul_tensor_15: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default_5, 1);  reciprocal_default_5 = None
        unsqueeze_default_40: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_15, -1);  convert_element_type_default_15 = None
        unsqueeze_default_41: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, -1);  unsqueeze_default_40 = None
        unsqueeze_default_42: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, -1);  mul_tensor_15 = None
        unsqueeze_default_43: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, -1);  unsqueeze_default_42 = None
        sub_tensor_5: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_default_41);  convolution_93 = unsqueeze_default_41 = None
        mul_tensor_16: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_43);  sub_tensor_5 = unsqueeze_default_43 = None
        unsqueeze_default_44: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
        unsqueeze_default_45: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, -1);  unsqueeze_default_44 = None
        mul_tensor_17: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_16, unsqueeze_default_45);  mul_tensor_16 = unsqueeze_default_45 = None
        unsqueeze_default_46: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg470_1, -1);  arg470_1 = None
        unsqueeze_default_47: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, -1);  unsqueeze_default_46 = None
        add_tensor_11: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_17, unsqueeze_default_47);  mul_tensor_17 = unsqueeze_default_47 = None
        convert_element_type_default_17: "bf16[128, 192, 8, 8]" = torch.ops.prims.convert_element_type.default(add_tensor_11, torch.bfloat16);  add_tensor_11 = None
        relu_default_5: "bf16[128, 192, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_default_17);  convert_element_type_default_17 = None
        cat_default_2: "bf16[128, 2048, 8, 8]" = torch.ops.aten.cat.default([relu_default, cat_default, cat_default_1, relu_default_5], 1);  relu_default = cat_default = cat_default_1 = relu_default_5 = None
        mean_dim: "bf16[128, 2048, 1, 1]" = torch.ops.aten.mean.dim(cat_default_2, [-1, -2], True);  cat_default_2 = None
        reshape_default: "bf16[128, 2048]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        permute_default: "bf16[2048, 1000]" = torch.ops.aten.permute.default(arg471_1, [1, 0]);  arg471_1 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([320], dtype=torch.bfloat16, device='cuda'),
    torch.randn([320], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 320, 8, 8], dtype=torch.bfloat16, device='cuda'),
    torch.randn([320], dtype=torch.bfloat16, device='cuda'),
    torch.randn([320], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 384, 8, 8], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 384, 8, 8], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 384, 8, 8], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 384, 8, 8], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([384], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 192, 8, 8], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1000, 2048], dtype=torch.bfloat16, device='cuda'),
    [128, 2048],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
