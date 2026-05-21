"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: fa555e0728c5
Shape hash: 5ef6542c
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
_shapes_config = "(T([32768, 512], f32), T([256, 128, 512], b8), T([], f32), S([256, 128, 512]), S([32768, 512]), S([512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_708: "f32[32768, 512]", arg1313_1: "b8[256, 128, 512]", full_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 512]" = torch.ops.aten.view.default(mm_708, _shape_param_0);  mm_708 = _shape_param_0 = None
        where_self: "f32[256, 128, 512]" = torch.ops.aten.where.self(arg1313_1, full_1, view_default);  arg1313_1 = full_1 = view_default = None
        view_default_1: "f32[32768, 512]" = torch.ops.aten.view.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        permute_default: "f32[512, 32768]" = torch.ops.aten.permute.default(view_default_1, [1, 0])
        sum_dim_int_list: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True);  view_default_1 = None
        view_default_2: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_2);  sum_dim_int_list = _shape_param_2 = None
        return (permute_default, view_default_2)



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
