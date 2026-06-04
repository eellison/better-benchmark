"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer_005
Pattern hash: 869474330b72
Shape hash: b953a7f6
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([768, 64, 64], f16), S([1, 12, 64, 64, 64]), S([1, 12, 4096, 64]), S([1, 4096, 768]), S([4096, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_11: "f16[768, 64, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_0);  bmm_11 = _shape_param_0 = None
        view_default_1: "f16[1, 12, 4096, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        clone_default: "f16[1, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_2: "f16[1, 4096, 768]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        view_default_3: "f16[4096, 768]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        return view_default_3

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
