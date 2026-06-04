"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train_000
Pattern hash: c3360ba4828a
Shape hash: 794de873
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 256, 13, 13], f32), T([512, 256, 13, 13], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_23: "f32[512, 256, 13, 13]", convolution_24: "f32[512, 256, 13, 13]"):
        # No stacktrace found for following nodes
        relu_default: "f32[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        relu_default_1: "f32[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
        cat_default: "f32[512, 512, 13, 13]" = torch.ops.aten.cat.default([relu_default, relu_default_1], 1)
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[512, 512, 13, 13]" = torch.ops.prims.inductor_random.default([512, 512, 13, 13], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[512, 512, 13, 13]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul_tensor: "f32[512, 512, 13, 13]" = torch.ops.aten.mul.Tensor(gt_scalar, cat_default);  gt_scalar = cat_default = None
        mul_tensor_1: "f32[512, 512, 13, 13]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        le_scalar: "b8[512, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        le_scalar_1: "b8[512, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (mul_tensor_1, le_scalar, le_scalar_1)

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
