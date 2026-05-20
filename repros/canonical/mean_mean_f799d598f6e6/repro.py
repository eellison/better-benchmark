"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-8-9-linux.aws.h100_graph74
Pattern hash: f799d598f6e6
Shape hash: 98be09ee
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1, 256, 64, 64], f64), T([256], f64), T([256], f64))"

class Repro(torch.nn.Module):
    def forward(self, convolution_2: "f64[1, 256, 64, 64]", arg456_1: "f64[256]", arg457_1: "f64[256]"):
        # No stacktrace found for following nodes
        mean_dim: "f64[1, 1, 64, 64]" = torch.ops.aten.mean.dim(convolution_2, [-3], True)
        sub_tensor: "f64[1, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_2, mean_dim)
        pow_tensor_scalar: "f64[1, 256, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(sub_tensor, 2);  sub_tensor = None
        mean_dim_1: "f64[1, 1, 64, 64]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-3], True);  pow_tensor_scalar = None
        sub_tensor_1: "f64[1, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_2, mean_dim);  convolution_2 = mean_dim = None
        add_tensor: "f64[1, 1, 64, 64]" = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06);  mean_dim_1 = None
        sqrt_default: "f64[1, 1, 64, 64]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        div_tensor: "f64[1, 256, 64, 64]" = torch.ops.aten.div.Tensor(sub_tensor_1, sqrt_default);  sub_tensor_1 = sqrt_default = None
        unsqueeze_default: "f64[256, 1]" = torch.ops.aten.unsqueeze.default(arg456_1, 1);  arg456_1 = None
        unsqueeze_default_1: "f64[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        mul_tensor: "f64[1, 256, 64, 64]" = torch.ops.aten.mul.Tensor(unsqueeze_default_1, div_tensor);  unsqueeze_default_1 = div_tensor = None
        unsqueeze_default_2: "f64[256, 1]" = torch.ops.aten.unsqueeze.default(arg457_1, 1);  arg457_1 = None
        unsqueeze_default_3: "f64[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 2);  unsqueeze_default_2 = None
        add_tensor_1: "f64[1, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_tensor, unsqueeze_default_3);  mul_tensor = unsqueeze_default_3 = None
        return add_tensor_1


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
