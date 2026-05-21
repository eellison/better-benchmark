"""
Standalone repro captured via capture_hook.
Label: timm_timm_vit_base_patch16_siglip_256_infer_infer_000
Pattern hash: dcfc5ad229c7
Shape hash: 7b8c05ab
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
_shapes_config = "(T([32768, 1536], f32), S([128, 256, 1536]), S([128, 256, 2, 12, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_48: "f32[32768, 1536]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 256, 1536]" = torch.ops.aten.view.default(addmm_48, _shape_param_0);  addmm_48 = _shape_param_0 = None
        view_default_1: "f32[128, 256, 2, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f32[2, 128, 12, 256, 64]" = torch.ops.aten.permute.default(view_default_1, [2, 0, 3, 1, 4]);  view_default_1 = None
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[128, 12, 256, 64]" = unbind_int[0]
        getitem_1: "f32[128, 12, 256, 64]" = unbind_int[1];  unbind_int = None
        return (getitem, getitem_1)



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
