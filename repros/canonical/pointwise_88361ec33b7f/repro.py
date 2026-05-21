"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer_007
Pattern hash: 88361ec33b7f
Shape hash: d0baddd1
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
_shapes_config = "(T([32, 1, 1], f32), T([32, 32000], f32), S([32, 1]), S([32]), S([32, 1, 32000]), S([32, 32000]))"

class Repro(torch.nn.Module):
    def forward(self, bmm: "f32[32, 1, 1]", mm: "f32[32, 32000]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[32, 1]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute_default: "f32[32, 1]" = torch.ops.aten.permute.default(view_default, [0, 1]);  view_default = None
        view_default_1: "f32[32]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        unsqueeze_default: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(view_default_1, -1);  view_default_1 = None
        unsqueeze_default_1: "f32[1, 32, 32000]" = torch.ops.aten.unsqueeze.default(mm, 0);  mm = None
        view_default_2: "f32[32, 1, 32000]" = torch.ops.aten.view.default(unsqueeze_default_1, _shape_param_2);  unsqueeze_default_1 = _shape_param_2 = None
        permute_default_1: "f32[32, 32000, 1]" = torch.ops.aten.permute.default(view_default_2, [0, 2, 1]);  view_default_2 = None
        view_default_3: "f32[32, 32000]" = torch.ops.aten.view.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None
        cat_default: "f32[32, 32001]" = torch.ops.aten.cat.default([unsqueeze_default, view_default_3], 1);  unsqueeze_default = view_default_3 = None
        div_tensor: "f32[32, 32001]" = torch.ops.aten.div.Tensor(cat_default, 0.07);  cat_default = None
        return div_tensor



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
