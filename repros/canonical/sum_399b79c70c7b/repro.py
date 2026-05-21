"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train_001
Pattern hash: 399b79c70c7b
Shape hash: 98c2700a
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
_shapes_config = "(T([512, 16, 55, 55], f32), T([512, 16, 55, 55], f32), T([512, 16, 55, 55], f32), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_66: "f32[512, 16, 55, 55]", getitem_69: "f32[512, 16, 55, 55]", arg29_1: "f32[512, 16, 55, 55]", full: "f32[]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[512, 16, 55, 55]" = torch.ops.aten.add.Tensor(getitem_66, getitem_69);  getitem_66 = getitem_69 = None
        le_scalar: "b8[512, 16, 55, 55]" = torch.ops.aten.le.Scalar(arg29_1, 0);  arg29_1 = None
        where_self: "f32[512, 16, 55, 55]" = torch.ops.aten.where.self(le_scalar, full, add_tensor);  le_scalar = full = add_tensor = None
        sum_dim_int_list: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
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
