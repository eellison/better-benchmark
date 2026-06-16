"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_infer
Pattern hash: 76ffbcc3b8af
Shape hash: a8ee30c6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 256, 13, 13]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        relu: "bf16[128, 256, 13, 13]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, _shape_param_0, _shape_param_1, [0, 0], [1, 1], False);  relu = _shape_param_0 = _shape_param_1 = None
        getitem: "bf16[128, 256, 6, 6]" = _low_memory_max_pool_with_offsets[0]
        getitem_1: "i8[128, 256, 6, 6]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = getitem_1 = None
        _adaptive_avg_pool2d: "bf16[128, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem, [6, 6]);  getitem = None
        view: "bf16[128, 9216]" = torch.ops.aten.view.default(_adaptive_avg_pool2d, _shape_param_2);  _adaptive_avg_pool2d = _shape_param_2 = None
        return view



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
