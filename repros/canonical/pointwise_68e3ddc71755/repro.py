"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-6-7-linux.aws.h100_graph58
Pattern hash: 68e3ddc71755
Shape hash: 5bca385a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 720], bf16), S([32, 16, 720]), S([32, 16, 3, 4, 60]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_32: "bf16[512, 720]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "bf16[32, 16, 720]" = torch.ops.aten.view.default(addmm_32, _shape_param_0);  addmm_32 = _shape_param_0 = None
        view_default_1: "bf16[32, 16, 3, 4, 60]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "bf16[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_default_1, [2, 0, 3, 1, 4]);  view_default_1 = None
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "bf16[32, 4, 16, 60]" = unbind_int[0]
        getitem_1: "bf16[32, 4, 16, 60]" = unbind_int[1]
        getitem_2: "bf16[32, 4, 16, 60]" = unbind_int[2];  unbind_int = None
        constant_pad_nd_default: "bf16[32, 4, 16, 64]" = torch.ops.aten.constant_pad_nd.default(getitem, [0, 4], 0.0);  getitem = None
        constant_pad_nd_default_1: "bf16[32, 4, 16, 64]" = torch.ops.aten.constant_pad_nd.default(getitem_1, [0, 4], 0.0);  getitem_1 = None
        constant_pad_nd_default_2: "bf16[32, 4, 16, 64]" = torch.ops.aten.constant_pad_nd.default(getitem_2, [0, 4], 0.0);  getitem_2 = None
        return (constant_pad_nd_default, constant_pad_nd_default_1, constant_pad_nd_default_2)


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
