"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_nfnet_infer_000
Pattern hash: d89eb45ec337
Shape hash: a50830a5
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
_shapes_config = "(T([128, 1536, 1, 1], f16), T([128, 1536, 12, 12], f16), T([], f16), T([128, 1536, 12, 12], f16))"

class Repro(torch.nn.Module):
    def forward(self, convolution_60: "f16[128, 1536, 1, 1]", convolution_58: "f16[128, 1536, 12, 12]", arg174_1: "f16[]", add_81: "f16[128, 1536, 12, 12]"):
        # No stacktrace found for following nodes
        sigmoid_default: "f16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60);  convolution_60 = None
        mul_tensor: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_58, sigmoid_default);  convolution_58 = sigmoid_default = None
        mul_tensor_1: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        mul_tensor_2: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg174_1);  mul_tensor_1 = arg174_1 = None
        mul_tensor_3: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.2);  mul_tensor_2 = None
        add_tensor: "f16[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_tensor_3, add_81);  mul_tensor_3 = add_81 = None
        convert_element_type_default: "f32[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        mul_tensor_4: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_5: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[128, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_tensor_5);  mul_tensor_5 = None
        add_tensor_1: "f32[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_6: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_1);  mul_tensor_4 = add_tensor_1 = None
        convert_element_type_default_1: "f16[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.float16);  mul_tensor_6 = None
        mul_tensor_7: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.7015043497085571);  convert_element_type_default_1 = None
        mul_tensor_8: "f16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.8980265101338745);  mul_tensor_7 = None
        avg_pool2d_default: "f16[128, 1536, 6, 6]" = torch.ops.aten.avg_pool2d.default(mul_tensor_8, [2, 2], [2, 2], [0, 0], True, False);  mul_tensor_8 = None
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
