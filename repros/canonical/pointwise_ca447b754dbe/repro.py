"""
Standalone repro captured via capture_hook.
Label: timm_timm_visformer_small_infer_infer_000
Pattern hash: ca447b754dbe
Shape hash: 109ec8bf
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([768, 49, 128], f32), S([128, 6, 49, 128]), S([128, 768, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_15: "f32[768, 49, 128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 6, 49, 128]" = torch.ops.aten.view.default(bmm_15, _shape_param_0);  bmm_15 = _shape_param_0 = None
        permute_default: "f32[128, 6, 128, 49]" = torch.ops.aten.permute.default(view_default, [0, 1, 3, 2]);  view_default = None
        clone_default: "f32[128, 6, 128, 49]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1: "f32[128, 768, 7, 7]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        return view_default_1

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
