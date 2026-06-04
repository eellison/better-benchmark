"""
Standalone repro captured via capture_hook.
Label: timm_timm_vit_base_patch14_dinov2.lvd142m_train_train_001
Pattern hash: c6dd83a0bedb
Shape hash: 360a332a
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 12, 1370, 64], f32), S([175360, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg107_1: "f32[128, 12, 1370, 64]", _shape_param_0):
        # No stacktrace found for following nodes
        permute_default: "f32[128, 1370, 12, 64]" = torch.ops.aten.permute.default(arg107_1, [0, 2, 1, 3]);  arg107_1 = None
        clone_default: "f32[128, 1370, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        _unsafe_view_default: "f32[128, 1370, 768]" = torch.ops.aten._unsafe_view.default(clone_default, [128, 1370, 768]);  clone_default = None
        view_default: "f32[175360, 768]" = torch.ops.aten.view.default(_unsafe_view_default, _shape_param_0);  _unsafe_view_default = _shape_param_0 = None
        return view_default

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
