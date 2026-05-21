"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_infer_000
Pattern hash: 0a306c604828
Shape hash: 59e61e38
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
_shapes_config = "(T([512, 256, 13, 13], f16), T([512, 256, 13, 13], f16))"

class Repro(torch.nn.Module):
    def forward(self, convolution_23: "f16[512, 256, 13, 13]", convolution_24: "f16[512, 256, 13, 13]"):
        # No stacktrace found for following nodes
        relu_default: "f16[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        relu_default_1: "f16[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
        cat_default: "f16[512, 512, 13, 13]" = torch.ops.aten.cat.default([relu_default, relu_default_1], 1);  relu_default = relu_default_1 = None
        return cat_default



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
