"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_infer_000
Pattern hash: 0de8974bd5cb
Shape hash: c2abb250
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 480], f32), S([512, 16, 480]), S([8192, 480]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_34: "f32[8192, 480]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[512, 16, 480]" = torch.ops.aten.view.default(addmm_34, _shape_param_0);  addmm_34 = _shape_param_0 = None
        neg_default: "f32[512, 16, 480]" = torch.ops.aten.neg.default(view_default)
        exp_default: "f32[512, 16, 480]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[512, 16, 480]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[512, 16, 480]" = torch.ops.aten.div.Tensor(view_default, add_tensor);  view_default = add_tensor = None
        view_default_1: "f32[8192, 480]" = torch.ops.aten.view.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
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
