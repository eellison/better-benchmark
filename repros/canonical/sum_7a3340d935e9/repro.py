"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_005
Pattern hash: 7a3340d935e9
Shape hash: 965c7f5d
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
_shapes_config = "(T([8, 1024, 768], f32), T([8, 1024, 768], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg303_1: "f32[8, 1024, 768]", arg230_1: "f32[8, 1024, 768]"):
        # No stacktrace found for following nodes
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg303_1, arg230_1);  arg303_1 = arg230_1 = None
        sum_dim_int_list: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
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
