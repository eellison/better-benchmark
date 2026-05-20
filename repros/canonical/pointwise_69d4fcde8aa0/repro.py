"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf-3-6-linux.aws.a100_graph16
Pattern hash: 69d4fcde8aa0
Shape hash: 75902420
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([320], bf16), T([320], bf16), T([128, 320, 7, 7], bf16), T([320], bf16), T([320], bf16))"

class Repro(torch.nn.Module):
    def forward(self, arg252_1: "bf16[320]", arg253_1: "bf16[320]", convolution_50: "bf16[128, 320, 7, 7]", arg254_1: "bf16[320]", arg255_1: "bf16[320]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[320]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.float32);  arg252_1 = None
        convert_element_type_default_1: "f32[320]" = torch.ops.prims.convert_element_type.default(arg253_1, torch.float32);  arg253_1 = None
        add_tensor: "f32[320]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[320]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        sub_tensor: "f32[128, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_default_1);  convolution_50 = unsqueeze_default_1 = None
        mul_tensor_1: "f32[128, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "bf16[320, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_default_5: "bf16[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 320, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "bf16[320, 1]" = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_default_7: "bf16[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 320, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "bf16[128, 320, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        return convert_element_type_default_2


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
