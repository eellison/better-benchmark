"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-8-9-linux.aws.h100_graph74
Pattern hash: 79f7d0111cf0
Shape hash: 691522b0
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 5120], f64), T([1280, 5120], f64), S([1, 64, 64, 5120]), S([4096, 5120]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_126: "f64[4096, 5120]", arg450_1: "f64[1280, 5120]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f64[1, 64, 64, 5120]" = torch.ops.aten.view.default(addmm_126, _shape_param_0);  addmm_126 = _shape_param_0 = None
        mul_tensor: "f64[1, 64, 64, 5120]" = torch.ops.aten.mul.Tensor(view_default, 0.5)
        mul_tensor_1: "f64[1, 64, 64, 5120]" = torch.ops.aten.mul.Tensor(view_default, 0.7071067811865476);  view_default = None
        erf_default: "f64[1, 64, 64, 5120]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f64[1, 64, 64, 5120]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f64[1, 64, 64, 5120]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        view_default_1: "f64[4096, 5120]" = torch.ops.aten.view.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        permute_default: "f64[5120, 1280]" = torch.ops.aten.permute.default(arg450_1, [1, 0]);  arg450_1 = None
        return (view_default_1, permute_default)


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
