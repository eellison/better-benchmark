"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_infer_000
Pattern hash: f9c1d1b08ddb
Shape hash: 42fc6e04
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
_shapes_config = "(T([512, 128, 27, 27], f16), T([512, 128, 27, 27], f16))"

class Repro(torch.nn.Module):
    def forward(self, convolution_11: "f16[512, 128, 27, 27]", convolution_12: "f16[512, 128, 27, 27]"):
        # No stacktrace found for following nodes
        relu_default: "f16[512, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        relu_default_1: "f16[512, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        cat_default: "f16[512, 256, 27, 27]" = torch.ops.aten.cat.default([relu_default, relu_default_1], 1);  relu_default = relu_default_1 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_default, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_default = None
        getitem: "f16[512, 256, 13, 13]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_1: "i8[512, 256, 13, 13]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        return (getitem, getitem_1)



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
