"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_train_002
Pattern hash: 7953a8b0bbba
Shape hash: 30e28b9a
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 256], f32), S([8, 1500, 256]), S([12000, 256]), S([256]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[8, 256]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[8, 1, 256]" = torch.ops.aten.unsqueeze.default(mm, 1);  mm = None
        expand_default: "f32[8, 1500, 256]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        div_scalar: "f32[8, 1500, 256]" = torch.ops.aten.div.Scalar(expand_default, 1500);  expand_default = None
        view_default: "f32[12000, 256]" = torch.ops.aten.view.default(div_scalar, _shape_param_1);  div_scalar = _shape_param_1 = None
        permute_default: "f32[256, 12000]" = torch.ops.aten.permute.default(view_default, [1, 0])
        sum_dim_int_list: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_default, [0], True);  view_default = None
        view_default_1: "f32[256]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_2);  sum_dim_int_list = _shape_param_2 = None
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
