"""
Standalone repro captured via capture_hook.
Label: timm_timm_visformer_small_train_train_001
Pattern hash: 88280aa05d1d
Shape hash: 75daa4cb
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
_shapes_config = "(T([768, 196, 64], f32), T([768, 64, 196], f32), T([768, 196, 64], f32), S([128, 6, 196, 64]), S([128, 6, 64, 196]), S([128, 6, 196, 64]), S([3, 128, 6, 196, 64]), S([128, 1152, 14, 14]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_28: "f32[768, 196, 64]", bmm_30: "f32[768, 64, 196]", bmm_31: "f32[768, 196, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[128, 6, 196, 64]" = torch.ops.aten.view.default(bmm_28, _shape_param_0);  bmm_28 = _shape_param_0 = None
        view_default_1: "f32[128, 6, 64, 196]" = torch.ops.aten.view.default(bmm_30, _shape_param_1);  bmm_30 = _shape_param_1 = None
        view_default_2: "f32[128, 6, 196, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_2);  bmm_31 = _shape_param_2 = None
        permute_default: "f32[128, 6, 196, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 1, 3, 2]);  view_default_1 = None
        cat_default: "f32[384, 6, 196, 64]" = torch.ops.aten.cat.default([view_default_2, permute_default, view_default]);  view_default_2 = permute_default = view_default = None
        view_default_3: "f32[3, 128, 6, 196, 64]" = torch.ops.aten.view.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None
        permute_default_1: "f32[128, 3, 6, 64, 196]" = torch.ops.aten.permute.default(view_default_3, [1, 0, 2, 4, 3]);  view_default_3 = None
        clone_default: "f32[128, 3, 6, 64, 196]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_4: "f32[128, 1152, 14, 14]" = torch.ops.aten.view.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
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
