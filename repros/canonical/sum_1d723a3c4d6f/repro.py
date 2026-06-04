"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_infer_000
Pattern hash: 1d723a3c4d6f
Shape hash: fb801ae1
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 5, 2, 426888], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 5, 2, 426888]"):
        # No stacktrace found for following nodes
        slice_tensor: "f32[8, 4, 2, 426888]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 1, 9223372036854775807);  arg0_1 = None
        slice_tensor_1: "f32[8, 4, 2, 382788]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 382788);  slice_tensor = None
        sum_dim_int_list: "f32[8, 2, 382788]" = torch.ops.aten.sum.dim_IntList(slice_tensor_1, [1]);  slice_tensor_1 = None
        return sum_dim_int_list

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
