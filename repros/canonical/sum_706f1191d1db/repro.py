"""
Standalone repro captured via capture_hook.
Label: timm_timm_vit_base_patch16_siglip_256_train_train_001
Pattern hash: 706f1191d1db
Shape hash: bec85e89
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
_shapes_config = "(T([128, 768], f32), S([128, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, arg280_1: "f32[128, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        full_default: "f32[128, 1, 768]" = torch.ops.aten.full.default([128, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[128, 1, 768]" = torch.ops.aten.select_scatter.default(full_default, arg280_1, 1, 0);  full_default = arg280_1 = None
        view_default: "f32[128, 768]" = torch.ops.aten.view.default(select_scatter_default, _shape_param_0);  select_scatter_default = _shape_param_0 = None
        permute_default: "f32[768, 128]" = torch.ops.aten.permute.default(view_default, [1, 0])
        sum_dim_int_list: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_default, [0], True);  view_default = None
        view_default_1: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        return (permute_default, view_default_1)



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
