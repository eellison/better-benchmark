"""
Standalone repro captured via capture_hook.
Label: torchbench_lennard_jones_infer
Pattern hash: 37c87a1bda68
Shape hash: ffb75246
"""
import sys
from pathlib import Path

import sys
from pathlib import Path
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 1]", arg2_1: "f32[1000, 1]", arg1_1: "f32[16]", arg3_1: "f32[16, 16]"):
        # No stacktrace found for following nodes
        permute_default: "f32[1, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mul_tensor: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(arg2_1, permute_default);  arg2_1 = permute_default = None
        mul_tensor_1: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(mul_tensor, 1);  mul_tensor = None
        mul_tensor_2: "f32[16]" = torch.ops.aten.mul.Tensor(arg1_1, 1);  arg1_1 = None
        add_tensor: "f32[1000, 16]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        tanh_default: "f32[1000, 16]" = torch.ops.aten.tanh.default(add_tensor);  add_tensor = None
        permute_default_1: "f32[16, 16]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        return (tanh_default, permute_default_1)


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
