"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer_009
Pattern hash: 7cc5c5d6f90f
Shape hash: 348b888e
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 32000], f32), T([32, 128], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[128, 32000]", arg0_1: "f32[32, 128]"):
        # No stacktrace found for following nodes
        slice_tensor: "f32[128, 32]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, 32)
        permute_default: "f32[128, 32]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        copy_default: "f32[128, 32]" = torch.ops.aten.copy.default(slice_tensor, permute_default);  slice_tensor = permute_default = None
        slice_scatter_default: "f32[128, 32000]" = torch.ops.aten.slice_scatter.default(arg1_1, copy_default, 1, 0, 32);  copy_default = None
        copy__default: "f32[128, 32000]" = torch.ops.aten.copy_.default(arg1_1, slice_scatter_default);  arg1_1 = slice_scatter_default = None
        return copy__default

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
