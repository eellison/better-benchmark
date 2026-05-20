"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf-2-6-linux.aws.a100_graph15
Pattern hash: 19128622cc10
Shape hash: 1f829619
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 3072, 8, 8], bf16), T([1000, 3072], bf16), S([128, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_80: "bf16[128, 3072, 8, 8]", arg232_1: "bf16[1000, 3072]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[128, 3072, 8, 8]" = torch.ops.prims.convert_element_type.default(convolution_80, torch.float32);  convolution_80 = None
        mul_tensor: "f32[128, 3072, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[128, 3072, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[128, 3072, 8, 8]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 3072, 8, 8]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 3072, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1: "bf16[128, 3072, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        mul_tensor_3: "bf16[128, 3072, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.7015043497085571);  convert_element_type_default_1 = None
        mean_dim: "bf16[128, 3072, 1, 1]" = torch.ops.aten.mean.dim(mul_tensor_3, [-1, -2], True);  mul_tensor_3 = None
        view_default: "bf16[128, 3072]" = torch.ops.aten.view.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        permute_default: "bf16[3072, 1000]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        return (view_default, permute_default)


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
