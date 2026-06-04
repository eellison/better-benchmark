"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train_013
Pattern hash: ca25949bc9c4
Shape hash: d4e37485
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 4096, 512], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 4096, 512]"):
        # No stacktrace found for following nodes
        slice_tensor: "f32[8, 4096, 256]" = torch.ops.aten.slice.Tensor(arg0_1, 2, 0, 256)
        slice_tensor_1: "f32[8, 4096, 256]" = torch.ops.aten.slice.Tensor(arg0_1, 2, 256, 512);  arg0_1 = None
        add_tensor: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
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
