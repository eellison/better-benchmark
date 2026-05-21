"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train_001
Pattern hash: 81ede6e15af0
Shape hash: 1860175b
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
_shapes_config = "(T([128, 12, 197, 64], f32, stride=(151296, 64, 768, 1)), T([128, 12, 197, 64], f32, stride=(151296, 64, 768, 1)), T([128, 12, 197, 64], f32, stride=(151296, 64, 768, 1)), S([3, 128, 12, 197, 64]), S([128, 197, 2304]), S([25216, 2304]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_44: "f32[128, 12, 197, 64]", getitem_45: "f32[128, 12, 197, 64]", getitem_46: "f32[128, 12, 197, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        cat_default: "f32[384, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_44, getitem_45, getitem_46]);  getitem_44 = getitem_45 = getitem_46 = None
        view_default: "f32[3, 128, 12, 197, 64]" = torch.ops.aten.view.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None
        permute_default: "f32[128, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_default, [1, 3, 0, 2, 4]);  view_default = None
        clone_default: "f32[128, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1: "f32[128, 197, 2304]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        view_default_2: "f32[25216, 2304]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        return view_default_2



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
