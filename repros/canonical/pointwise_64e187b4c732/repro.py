"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-1-7-linux.aws.h100_graph48
Pattern hash: 64e187b4c732
Shape hash: 1e3d3922
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([384], bf16), T([384], bf16), T([128, 384, 8, 8], bf16), T([384], bf16), T([384], bf16), T([384], bf16), T([384], bf16), T([128, 384, 8, 8], bf16), T([384], bf16), T([384], bf16), T([128, 2048, 8, 8], bf16))"

class Repro(torch.nn.Module):
    def forward(self, arg432_1: "bf16[384]", arg433_1: "bf16[384]", convolution_86: "bf16[128, 384, 8, 8]", arg434_1: "bf16[384]", arg435_1: "bf16[384]", arg452_1: "bf16[384]", arg453_1: "bf16[384]", convolution_90: "bf16[128, 384, 8, 8]", arg454_1: "bf16[384]", arg455_1: "bf16[384]", cat_11: "bf16[128, 2048, 8, 8]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[384]" = torch.ops.prims.convert_element_type.default(arg432_1, torch.float32);  arg432_1 = None
        convert_element_type_default_1: "f32[384]" = torch.ops.prims.convert_element_type.default(arg433_1, torch.float32);  arg433_1 = None
        add_tensor: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 0.001);  convert_element_type_default_1 = None
        sqrt_default: "f32[384]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        sub_tensor: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_default_1);  convolution_86 = unsqueeze_default_1 = None
        mul_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
        unsqueeze_default_5: "bf16[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(arg435_1, -1);  arg435_1 = None
        unsqueeze_default_7: "bf16[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        relu_default: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        convert_element_type_default_3: "f32[384]" = torch.ops.prims.convert_element_type.default(arg452_1, torch.float32);  arg452_1 = None
        convert_element_type_default_4: "f32[384]" = torch.ops.prims.convert_element_type.default(arg453_1, torch.float32);  arg453_1 = None
        add_tensor_2: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_default_4, 0.001);  convert_element_type_default_4 = None
        sqrt_default_1: "f32[384]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_8: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, -1);  convert_element_type_default_3 = None
        unsqueeze_default_9: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        sub_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_default_9);  convolution_90 = unsqueeze_default_9 = None
        mul_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_default_13: "bf16[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(arg455_1, -1);  arg455_1 = None
        unsqueeze_default_15: "bf16[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        convert_element_type_default_5: "bf16[128, 384, 8, 8]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        relu_default_1: "bf16[128, 384, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_default_5);  convert_element_type_default_5 = None
        avg_pool2d_default: "bf16[128, 2048, 8, 8]" = torch.ops.aten.avg_pool2d.default(cat_11, [3, 3], [1, 1], [1, 1]);  cat_11 = None
        return (relu_default, relu_default_1, avg_pool2d_default)


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
