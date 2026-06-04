"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer_004
Pattern hash: bcfec253e455
Shape hash: 586875ae
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128], f32), T([128], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg321_1: "f32[128]", arg322_1: "f32[128]"):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128]" = torch.ops.aten.mul.Tensor(arg321_1, 0.999)
        mul_tensor_1: "f32[128]" = torch.ops.aten.mul.Tensor(arg322_1, 0.0010000000000000009);  arg322_1 = None
        add_tensor: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        copy__default: "f32[128]" = torch.ops.aten.copy_.default(arg321_1, add_tensor);  arg321_1 = add_tensor = None
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
