"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train_000
Pattern hash: 69399b54497b
Shape hash: 8735f936
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 1536], f32), T([], f32), S([8, 512, 1536]), S([8, 512, 24, -1]), S([-1, 512, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_139: "f32[4096, 1536]", full: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[8, 512, 1536]" = torch.ops.aten.view.default(addmm_139, _shape_param_0);  addmm_139 = _shape_param_0 = None
        view_default_1: "f32[8, 512, 24, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        clone_default: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_2: "f32[192, 512, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        permute_default_1: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_default_2, [0, 2, 1]);  view_default_2 = None
        div_tensor: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_default_1, full);  permute_default_1 = full = None
        permute_default_2: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_tensor, [0, 2, 1]);  div_tensor = None
        return permute_default_2

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
