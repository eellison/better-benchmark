"""
Standalone repro captured via capture_hook.
Label: timm_timm_vit_base_patch16_siglip_256_infer_infer_000
Pattern hash: 7517fa77d424
Shape hash: b742dacb
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 768], f32), T([128, 1, 768], f32), S([128, 1, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_51: "f32[128, 768]", view_104: "f32[128, 1, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[128, 1, 768]" = torch.ops.aten.view.default(addmm_51, _shape_param_0);  addmm_51 = _shape_param_0 = None
        add_tensor: "f32[128, 1, 768]" = torch.ops.aten.add.Tensor(view_104, view_default);  view_104 = view_default = None
        select_int: "f32[128, 768]" = torch.ops.aten.select.int(add_tensor, 1, 0);  add_tensor = None
        return select_int

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
