"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_infer_000
Pattern hash: 5c93e9826aa8
Shape hash: 3f5b3c4b
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([320], f32), T([128, 320, 8, 8], f32), T([320], f32), T([320], f32), T([320], f32), T([384], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([192], f32), T([128, 192, 8, 8], f32), T([192], f32), T([192], f32), T([192], f32), S([128, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, arg427_1: "f32[320]", convolution_85: "f32[128, 320, 8, 8]", arg428_1: "f32[320]", arg429_1: "f32[320]", arg430_1: "f32[320]", arg437_1: "f32[384]", convolution_87: "f32[128, 384, 8, 8]", arg438_1: "f32[384]", arg439_1: "f32[384]", arg440_1: "f32[384]", arg442_1: "f32[384]", convolution_88: "f32[128, 384, 8, 8]", arg443_1: "f32[384]", arg444_1: "f32[384]", arg445_1: "f32[384]", arg457_1: "f32[384]", convolution_91: "f32[128, 384, 8, 8]", arg458_1: "f32[384]", arg459_1: "f32[384]", arg460_1: "f32[384]", arg462_1: "f32[384]", convolution_92: "f32[128, 384, 8, 8]", arg463_1: "f32[384]", arg464_1: "f32[384]", arg465_1: "f32[384]", arg467_1: "f32[192]", convolution_93: "f32[128, 192, 8, 8]", arg468_1: "f32[192]", arg469_1: "f32[192]", arg470_1: "f32[192]", _shape_param_0):
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg427_1, -1);  arg427_1 = None
        unsqueeze_default_1: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_default_1);  convolution_85 = unsqueeze_default_1 = None
        add_tensor: "f32[320]" = torch.ops.aten.add.Tensor(arg428_1, 0.001);  arg428_1 = None
        sqrt_default: "f32[320]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg429_1, -1);  arg429_1 = None
        unsqueeze_default_5: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
        unsqueeze_default_7: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        relu_default: "f32[128, 320, 8, 8]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        unsqueeze_default_8: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg437_1, -1);  arg437_1 = None
        unsqueeze_default_9: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_default_9);  convolution_87 = unsqueeze_default_9 = None
        add_tensor_2: "f32[384]" = torch.ops.aten.add.Tensor(arg438_1, 0.001);  arg438_1 = None
        sqrt_default_1: "f32[384]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg439_1, -1);  arg439_1 = None
        unsqueeze_default_13: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg440_1, -1);  arg440_1 = None
        unsqueeze_default_15: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        relu_default_1: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        unsqueeze_default_16: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg442_1, -1);  arg442_1 = None
        unsqueeze_default_17: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        sub_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_default_17);  convolution_88 = unsqueeze_default_17 = None
        add_tensor_4: "f32[384]" = torch.ops.aten.add.Tensor(arg443_1, 0.001);  arg443_1 = None
        sqrt_default_2: "f32[384]" = torch.ops.aten.sqrt.default(add_tensor_4);  add_tensor_4 = None
        reciprocal_default_2: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_default_2);  sqrt_default_2 = None
        mul_tensor_6: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_default_2, 1);  reciprocal_default_2 = None
        unsqueeze_default_18: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, -1);  mul_tensor_6 = None
        unsqueeze_default_19: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        mul_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_19);  sub_tensor_2 = unsqueeze_default_19 = None
        unsqueeze_default_20: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
        unsqueeze_default_21: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, -1);  unsqueeze_default_20 = None
        mul_tensor_8: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_21);  mul_tensor_7 = unsqueeze_default_21 = None
        unsqueeze_default_22: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg445_1, -1);  arg445_1 = None
        unsqueeze_default_23: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, -1);  unsqueeze_default_22 = None
        add_tensor_5: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_8, unsqueeze_default_23);  mul_tensor_8 = unsqueeze_default_23 = None
        relu_default_2: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_5);  add_tensor_5 = None
        cat_default: "f32[128, 768, 8, 8]" = torch.ops.aten.cat.default([relu_default_1, relu_default_2], 1);  relu_default_1 = relu_default_2 = None
        unsqueeze_default_24: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg457_1, -1);  arg457_1 = None
        unsqueeze_default_25: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, -1);  unsqueeze_default_24 = None
        sub_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_default_25);  convolution_91 = unsqueeze_default_25 = None
        add_tensor_6: "f32[384]" = torch.ops.aten.add.Tensor(arg458_1, 0.001);  arg458_1 = None
        sqrt_default_3: "f32[384]" = torch.ops.aten.sqrt.default(add_tensor_6);  add_tensor_6 = None
        reciprocal_default_3: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_default_3);  sqrt_default_3 = None
        mul_tensor_9: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_default_3, 1);  reciprocal_default_3 = None
        unsqueeze_default_26: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, -1);  mul_tensor_9 = None
        unsqueeze_default_27: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, -1);  unsqueeze_default_26 = None
        mul_tensor_10: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_27);  sub_tensor_3 = unsqueeze_default_27 = None
        unsqueeze_default_28: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg459_1, -1);  arg459_1 = None
        unsqueeze_default_29: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, -1);  unsqueeze_default_28 = None
        mul_tensor_11: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_10, unsqueeze_default_29);  mul_tensor_10 = unsqueeze_default_29 = None
        unsqueeze_default_30: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg460_1, -1);  arg460_1 = None
        unsqueeze_default_31: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, -1);  unsqueeze_default_30 = None
        add_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_11, unsqueeze_default_31);  mul_tensor_11 = unsqueeze_default_31 = None
        relu_default_3: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None
        unsqueeze_default_32: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg462_1, -1);  arg462_1 = None
        unsqueeze_default_33: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, -1);  unsqueeze_default_32 = None
        sub_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_default_33);  convolution_92 = unsqueeze_default_33 = None
        add_tensor_8: "f32[384]" = torch.ops.aten.add.Tensor(arg463_1, 0.001);  arg463_1 = None
        sqrt_default_4: "f32[384]" = torch.ops.aten.sqrt.default(add_tensor_8);  add_tensor_8 = None
        reciprocal_default_4: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_default_4);  sqrt_default_4 = None
        mul_tensor_12: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_default_4, 1);  reciprocal_default_4 = None
        unsqueeze_default_34: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, -1);  mul_tensor_12 = None
        unsqueeze_default_35: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, -1);  unsqueeze_default_34 = None
        mul_tensor_13: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_35);  sub_tensor_4 = unsqueeze_default_35 = None
        unsqueeze_default_36: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_default_37: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, -1);  unsqueeze_default_36 = None
        mul_tensor_14: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_13, unsqueeze_default_37);  mul_tensor_13 = unsqueeze_default_37 = None
        unsqueeze_default_38: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg465_1, -1);  arg465_1 = None
        unsqueeze_default_39: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, -1);  unsqueeze_default_38 = None
        add_tensor_9: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_14, unsqueeze_default_39);  mul_tensor_14 = unsqueeze_default_39 = None
        relu_default_4: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_9);  add_tensor_9 = None
        cat_default_1: "f32[128, 768, 8, 8]" = torch.ops.aten.cat.default([relu_default_3, relu_default_4], 1);  relu_default_3 = relu_default_4 = None
        unsqueeze_default_40: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg467_1, -1);  arg467_1 = None
        unsqueeze_default_41: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, -1);  unsqueeze_default_40 = None
        sub_tensor_5: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_default_41);  convolution_93 = unsqueeze_default_41 = None
        add_tensor_10: "f32[192]" = torch.ops.aten.add.Tensor(arg468_1, 0.001);  arg468_1 = None
        sqrt_default_5: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor_10);  add_tensor_10 = None
        reciprocal_default_5: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default_5);  sqrt_default_5 = None
        mul_tensor_15: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default_5, 1);  reciprocal_default_5 = None
        unsqueeze_default_42: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, -1);  mul_tensor_15 = None
        unsqueeze_default_43: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, -1);  unsqueeze_default_42 = None
        mul_tensor_16: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_43);  sub_tensor_5 = unsqueeze_default_43 = None
        unsqueeze_default_44: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
        unsqueeze_default_45: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, -1);  unsqueeze_default_44 = None
        mul_tensor_17: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_16, unsqueeze_default_45);  mul_tensor_16 = unsqueeze_default_45 = None
        unsqueeze_default_46: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg470_1, -1);  arg470_1 = None
        unsqueeze_default_47: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, -1);  unsqueeze_default_46 = None
        add_tensor_11: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_17, unsqueeze_default_47);  mul_tensor_17 = unsqueeze_default_47 = None
        relu_default_5: "f32[128, 192, 8, 8]" = torch.ops.aten.relu.default(add_tensor_11);  add_tensor_11 = None
        cat_default_2: "f32[128, 2048, 8, 8]" = torch.ops.aten.cat.default([relu_default, cat_default, cat_default_1, relu_default_5], 1);  relu_default = cat_default = cat_default_1 = relu_default_5 = None
        mean_dim: "f32[128, 2048, 1, 1]" = torch.ops.aten.mean.dim(cat_default_2, [-1, -2], True);  cat_default_2 = None
        as_strided_default: "f32[128, 2048, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 2048, 1, 1], [2048, 1, 2048, 2048]);  mean_dim = None
        view_default: "f32[128, 2048]" = torch.ops.aten.view.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None
        return view_default

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
