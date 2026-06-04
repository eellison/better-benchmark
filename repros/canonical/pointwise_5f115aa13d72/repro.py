"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_002
Pattern hash: 5f115aa13d72
Shape hash: 6f022578
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 768], f32), T([768], f32), S([1024, 8, 768]), S([1024, 8, 12, 64]), S([96, 1024, 64]), S([96, 2, 512, 64]), S([288, 64, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_45: "f32[8192, 768]", arg182_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[1024, 8, 768]" = torch.ops.aten.view.default(mm_45, _shape_param_0);  mm_45 = _shape_param_0 = None
        add_tensor: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_default, arg182_1);  view_default = arg182_1 = None
        view_default_1: "f32[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_default_1, [1, 0, 2, 3]);  view_default_1 = None
        permute_default_1: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default_2: "f32[96, 1024, 64]" = torch.ops.aten.view.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None
        view_default_3: "f32[96, 2, 512, 64]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        as_strided_default: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_default_3, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_default_3 = None
        unsqueeze_default: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        permute_default_2: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_default, [0, 1, 4, 2, 3]);  unsqueeze_default = None
        permute_default_3: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_default_2, [0, 1, 4, 3, 2]);  permute_default_2 = None
        clone_default: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_4: "f32[288, 64, 512]" = torch.ops.aten.view.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        permute_default_4: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_default_4, [0, 2, 1]);  view_default_4 = None
        return permute_default_4

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
