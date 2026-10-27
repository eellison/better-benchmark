"""
Standalone repro captured via capture_hook.
Label: alexnet
Pattern hash: d153e2bc6ddf
Shape hash: e0a58bd8
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
    def forward(self, arg0_1: "bf16[128, 256, 13, 13]", arg1_1: "bf16[4096, 9216]", arg2_1: "bf16[4096]", arg3_1: "bf16[4096, 4096]", arg4_1: "bf16[4096]", arg5_1: "bf16[1000, 4096]", arg6_1: "bf16[1000]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        relu: "bf16[128, 256, 13, 13]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, _shape_param_0, _shape_param_1, [0, 0], [1, 1], False);  relu = _shape_param_0 = _shape_param_1 = None
        getitem: "bf16[128, 256, 6, 6]" = _low_memory_max_pool_with_offsets[0]
        getitem_1: "i8[128, 256, 6, 6]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = getitem_1 = None
        _adaptive_avg_pool2d: "bf16[128, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem, [6, 6]);  getitem = None
        view: "bf16[128, 9216]" = torch.ops.aten.view.default(_adaptive_avg_pool2d, _shape_param_2);  _adaptive_avg_pool2d = _shape_param_2 = None
        permute: "bf16[9216, 4096]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg2_1, view, permute);  arg2_1 = view = permute = None
        relu_1: "bf16[128, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_1: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg4_1, relu_1, permute_1);  arg4_1 = relu_1 = permute_1 = None
        relu_2: "bf16[128, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_2: "bf16[4096, 1000]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_2: "bf16[128, 1000]" = torch.ops.aten.addmm.default(arg6_1, relu_2, permute_2);  arg6_1 = relu_2 = permute_2 = None
        return addmm_2



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
