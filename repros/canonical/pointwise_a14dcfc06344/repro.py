"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_infer_000
Pattern hash: a14dcfc06344
Shape hash: 225bfb21
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
_shapes_config = "(T([192], f32), T([128, 192, 17, 17], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([128, 192, 17, 17], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([128, 192, 17, 17], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([128, 192, 17, 17], f32), T([192], f32), T([192], f32), T([192], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg252_1: "f32[192]", convolution_50: "f32[128, 192, 17, 17]", arg253_1: "f32[192]", arg254_1: "f32[192]", arg255_1: "f32[192]", arg267_1: "f32[192]", convolution_53: "f32[128, 192, 17, 17]", arg268_1: "f32[192]", arg269_1: "f32[192]", arg270_1: "f32[192]", arg292_1: "f32[192]", convolution_58: "f32[128, 192, 17, 17]", arg293_1: "f32[192]", arg294_1: "f32[192]", arg295_1: "f32[192]", arg297_1: "f32[192]", convolution_59: "f32[128, 192, 17, 17]", arg298_1: "f32[192]", arg299_1: "f32[192]", arg300_1: "f32[192]"):
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_default_1);  convolution_50 = unsqueeze_default_1 = None
        add_tensor: "f32[192]" = torch.ops.aten.add.Tensor(arg253_1, 0.001);  arg253_1 = None
        sqrt_default: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_default_5: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_default_7: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        relu_default: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        unsqueeze_default_8: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_default_9: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_default_9);  convolution_53 = unsqueeze_default_9 = None
        add_tensor_2: "f32[192]" = torch.ops.aten.add.Tensor(arg268_1, 0.001);  arg268_1 = None
        sqrt_default_1: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_default_13: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_default_15: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        relu_default_1: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        unsqueeze_default_16: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_default_17: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        sub_tensor_2: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_default_17);  convolution_58 = unsqueeze_default_17 = None
        add_tensor_4: "f32[192]" = torch.ops.aten.add.Tensor(arg293_1, 0.001);  arg293_1 = None
        sqrt_default_2: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor_4);  add_tensor_4 = None
        reciprocal_default_2: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default_2);  sqrt_default_2 = None
        mul_tensor_6: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default_2, 1);  reciprocal_default_2 = None
        unsqueeze_default_18: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, -1);  mul_tensor_6 = None
        unsqueeze_default_19: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        mul_tensor_7: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_19);  sub_tensor_2 = unsqueeze_default_19 = None
        unsqueeze_default_20: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_default_21: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, -1);  unsqueeze_default_20 = None
        mul_tensor_8: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_21);  mul_tensor_7 = unsqueeze_default_21 = None
        unsqueeze_default_22: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_default_23: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, -1);  unsqueeze_default_22 = None
        add_tensor_5: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_8, unsqueeze_default_23);  mul_tensor_8 = unsqueeze_default_23 = None
        relu_default_2: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_5);  add_tensor_5 = None
        unsqueeze_default_24: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_default_25: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, -1);  unsqueeze_default_24 = None
        sub_tensor_3: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_default_25);  convolution_59 = unsqueeze_default_25 = None
        add_tensor_6: "f32[192]" = torch.ops.aten.add.Tensor(arg298_1, 0.001);  arg298_1 = None
        sqrt_default_3: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor_6);  add_tensor_6 = None
        reciprocal_default_3: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default_3);  sqrt_default_3 = None
        mul_tensor_9: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default_3, 1);  reciprocal_default_3 = None
        unsqueeze_default_26: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, -1);  mul_tensor_9 = None
        unsqueeze_default_27: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, -1);  unsqueeze_default_26 = None
        mul_tensor_10: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_27);  sub_tensor_3 = unsqueeze_default_27 = None
        unsqueeze_default_28: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_default_29: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, -1);  unsqueeze_default_28 = None
        mul_tensor_11: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_10, unsqueeze_default_29);  mul_tensor_10 = unsqueeze_default_29 = None
        unsqueeze_default_30: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_default_31: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, -1);  unsqueeze_default_30 = None
        add_tensor_7: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_11, unsqueeze_default_31);  mul_tensor_11 = unsqueeze_default_31 = None
        relu_default_3: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None
        cat_default: "f32[128, 768, 17, 17]" = torch.ops.aten.cat.default([relu_default, relu_default_1, relu_default_2, relu_default_3], 1);  relu_default = relu_default_1 = relu_default_2 = relu_default_3 = None
        avg_pool2d_default: "f32[128, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat_default, [3, 3], [1, 1], [1, 1]);  cat_default = None
        return avg_pool2d_default



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
