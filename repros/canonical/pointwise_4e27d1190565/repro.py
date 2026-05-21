"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_infer_000
Pattern hash: 4e27d1190565
Shape hash: 34d41de6
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
_shapes_config = "(T([1024, 192, 27, 27], f16))"

class Repro(torch.nn.Module):
    def forward(self, convolution_1: "f16[1024, 192, 27, 27]"):
        # No stacktrace found for following nodes
        relu_default: "f16[1024, 192, 27, 27]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_default = None
        getitem: "f16[1024, 192, 13, 13]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_1: "i8[1024, 192, 13, 13]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        return (getitem_1, getitem)



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
