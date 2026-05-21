"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train_003
Pattern hash: ad4468aef9e0
Shape hash: 66ab4406
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
_shapes_config = "(T([32, 12, 512, 64], f32, stride=(393216, 64, 768, 1)), T([32, 12, 512, 64], f32, stride=(393216, 64, 768, 1)), T([32, 12, 512, 64], f32, stride=(393216, 64, 768, 1)), S([32, 512, 768]), S([32, 512, 768]), S([32, 512, 768]), S([16384, 2304]), S([2304]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_20: "f32[32, 12, 512, 64]", getitem_22: "f32[32, 12, 512, 64]", getitem_21: "f32[32, 12, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        permute_default: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3]);  getitem_20 = None
        view_default: "f32[32, 512, 768]" = torch.ops.aten.view.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        permute_default_1: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_22, [0, 2, 1, 3]);  getitem_22 = None
        view_default_1: "f32[32, 512, 768]" = torch.ops.aten.view.default(permute_default_1, _shape_param_1);  permute_default_1 = _shape_param_1 = None
        permute_default_2: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
        view_default_2: "f32[32, 512, 768]" = torch.ops.aten.view.default(permute_default_2, _shape_param_2);  permute_default_2 = _shape_param_2 = None
        cat_default: "f32[32, 512, 2304]" = torch.ops.aten.cat.default([view_default, view_default_2, view_default_1], 2);  view_default = view_default_2 = view_default_1 = None
        view_default_3: "f32[16384, 2304]" = torch.ops.aten.view.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None
        sum_dim_int_list: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True);  view_default_3 = None
        view_default_4: "f32[2304]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_4);  sum_dim_int_list = _shape_param_4 = None
        return view_default_4



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
