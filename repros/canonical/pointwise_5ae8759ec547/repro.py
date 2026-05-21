"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train_000
Pattern hash: 5ae8759ec547
Shape hash: a59bcdb6
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
_shapes_config = "(T([512, 128, 27, 27], f32), T([512, 128, 27, 27], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_11: "f32[512, 128, 27, 27]", convolution_12: "f32[512, 128, 27, 27]"):
        # No stacktrace found for following nodes
        relu_default: "f32[512, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        relu_default_1: "f32[512, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        cat_default: "f32[512, 256, 27, 27]" = torch.ops.aten.cat.default([relu_default, relu_default_1], 1)
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_default, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_default = None
        getitem: "f32[512, 256, 13, 13]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_1: "i8[512, 256, 13, 13]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        le_scalar: "b8[512, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        le_scalar_1: "b8[512, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (getitem, getitem_1, le_scalar, le_scalar_1)



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
