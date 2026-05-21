"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train_000
Pattern hash: 2958fe7f4e37
Shape hash: 04355002
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
_shapes_config = "(T([512, 1280, 1, 1], f32), S([512, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_94: "f32[512, 1280, 1, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        relu_default: "f32[512, 1280, 1, 1]" = torch.ops.aten.relu.default(convolution_94);  convolution_94 = None
        view_default: "f32[512, 1280]" = torch.ops.aten.view.default(relu_default, _shape_param_0);  _shape_param_0 = None
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[512, 1280]" = torch.ops.prims.inductor_random.default([512, 1280], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.2);  inductor_random_default = None
        mul_tensor: "f32[512, 1280]" = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_1: "f32[512, 1280]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.25);  mul_tensor = None
        le_scalar: "b8[512, 1280, 1, 1]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (le_scalar, mul_tensor_1)



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
