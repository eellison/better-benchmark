"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_infer_005
Pattern hash: a80f6127c2d6
Shape hash: db9e7031
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 50272], f16), T([1, 50265], f16), S([1, 512, 50265]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f16[512, 50272]", arg2_1: "f16[1, 50265]", _shape_param_0):
        # No stacktrace found for following nodes
        slice_tensor: "f16[512, 50265]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -7);  mm = None
        view_default: "f16[1, 512, 50265]" = torch.ops.aten.view.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None
        add_tensor: "f16[1, 512, 50265]" = torch.ops.aten.add.Tensor(view_default, arg2_1);  view_default = arg2_1 = None
        return add_tensor

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
