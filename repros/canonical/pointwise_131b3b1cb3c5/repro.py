"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_infer_000
Pattern hash: 131b3b1cb3c5
Shape hash: 9d97ed0b
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
_shapes_config = "(T([1024, 256, 13, 13], f16), S([1024, 9216]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_4: "f16[1024, 256, 13, 13]", _shape_param_0):
        # No stacktrace found for following nodes
        relu_default: "f16[1024, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_default = None
        getitem: "f16[1024, 256, 6, 6]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_1: "i8[1024, 256, 6, 6]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        _adaptive_avg_pool2d_default: "f16[1024, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem, [6, 6]);  getitem = None
        view_default: "f16[1024, 9216]" = torch.ops.aten.view.default(_adaptive_avg_pool2d_default, _shape_param_0);  _adaptive_avg_pool2d_default = _shape_param_0 = None
        return (getitem_1, view_default)



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
