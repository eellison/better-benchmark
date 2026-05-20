"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-4-6-linux.aws.a100_graph24
Pattern hash: 1dccacf62804
Shape hash: 1027396c
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([32, 320, 7, 7], f16), T([320], f32), T([320], f32), T([1280, 320, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_50: "f16[32, 320, 7, 7]", arg305_1: "f32[320]", arg306_1: "f32[320]", arg307_1: "f32[1280, 320, 1, 1]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[32, 320, 7, 7]" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 320, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 320, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_1);  convolution_50 = getitem_1 = None
        mul_tensor: "f32[32, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_default_1: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 320, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_default_3: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[32, 320, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        convert_element_type_default_1: "f16[32, 320, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        convert_element_type_default_2: "f16[1280, 320, 1, 1]" = torch.ops.prims.convert_element_type.default(arg307_1, torch.float16);  arg307_1 = None
        return (convert_element_type_default_1, convert_element_type_default_2)


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
