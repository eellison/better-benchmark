"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_nfnet_train_000
Pattern hash: b1d8f8a4704b
Shape hash: be444ee4
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 3072, 6, 6], f32), S([128, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_80: "f32[128, 3072, 6, 6]", _shape_param_0):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, 0.5)
        mul_tensor_1: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, 0.7071067811865476);  convolution_80 = None
        erf_default: "f32[128, 3072, 6, 6]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 3072, 6, 6]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        mul_tensor_3: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.7015043497085571);  mul_tensor_2 = None
        mean_dim: "f32[128, 3072, 1, 1]" = torch.ops.aten.mean.dim(mul_tensor_3, [-1, -2], True);  mul_tensor_3 = None
        view_default: "f32[128, 3072]" = torch.ops.aten.view.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
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
