"""
Standalone repro captured via capture_hook.
Label: torchbench_lennard_jones_train
Pattern hash: 22dd6590f554
Shape hash: 45cd6de0
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
_shapes_config = "(T([1000, 16], f32), T([1000, 16], f32), S([16]))"

class Repro(torch.nn.Module):
    def forward(self, tanh: "f32[1000, 16]", mm_5: "f32[1000, 16]", _shape_param_0):
        # No stacktrace found for following nodes
        mul_tensor: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_tensor: "f32[1000, 16]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_1: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(mm_5, sub_tensor);  mm_5 = sub_tensor = None
        permute_default: "f32[16, 1000]" = torch.ops.aten.permute.default(mul_tensor_1, [1, 0])
        sum_dim_int_list: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0], True);  mul_tensor_1 = None
        reshape_default: "f32[16]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        return (permute_default, reshape_default)



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
