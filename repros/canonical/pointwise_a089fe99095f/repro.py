"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train_002
Pattern hash: a089fe99095f
Shape hash: f2847659
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 768], f32), T([768], f32), S([1024, 2, 768]), S([1024, 2, 12, 64]), S([24, 1024, 64]), S([96, 768, 64]))"

class Repro(torch.nn.Module):
    def forward(self, mm_46: "f32[2048, 768]", arg184_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[1024, 2, 768]" = torch.ops.aten.view.default(mm_46, _shape_param_0);  mm_46 = _shape_param_0 = None
        add_tensor: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_default, arg184_1);  view_default = arg184_1 = None
        view_default_1: "f32[1024, 2, 12, 64]" = torch.ops.aten.view.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_default_1, [1, 0, 2, 3]);  view_default_1 = None
        permute_default_1: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default_2: "f32[24, 1024, 64]" = torch.ops.aten.view.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None
        constant_pad_nd_default: "f32[24, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_default_2, [0, 0, 256, 256], -1.0);  view_default_2 = None
        as_strided_default: "f32[24, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_default, [24, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_default = None
        unsqueeze_default: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        clone_default: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default, memory_format = torch.contiguous_format);  unsqueeze_default = None
        view_default_3: "f32[96, 768, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        permute_default_2: "f32[96, 64, 768]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1]);  view_default_3 = None
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
