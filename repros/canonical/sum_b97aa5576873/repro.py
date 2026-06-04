"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_001
Pattern hash: b97aa5576873
Shape hash: f4ac7cfe
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32768, 49, 32], f32), T([32768, 32, 49], f32), T([32768, 49, 32], f32), S([8192, 4, 49, 32]), S([8192, 4, 32, 49]), S([8192, 4, 49, 32]), S([3, 8192, 4, 49, 32]), S([8192, 49, 384]), S([401408, 384]), S([384]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_92: "f32[32768, 49, 32]", bmm_94: "f32[32768, 32, 49]", bmm_95: "f32[32768, 49, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view_default: "f32[8192, 4, 49, 32]" = torch.ops.aten.view.default(bmm_92, _shape_param_0);  bmm_92 = _shape_param_0 = None
        view_default_1: "f32[8192, 4, 32, 49]" = torch.ops.aten.view.default(bmm_94, _shape_param_1);  bmm_94 = _shape_param_1 = None
        view_default_2: "f32[8192, 4, 49, 32]" = torch.ops.aten.view.default(bmm_95, _shape_param_2);  bmm_95 = _shape_param_2 = None
        permute_default: "f32[8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_default_1, [0, 1, 3, 2]);  view_default_1 = None
        mul_tensor: "f32[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(view_default_2, 0.1767766952966369);  view_default_2 = None
        cat_default: "f32[24576, 4, 49, 32]" = torch.ops.aten.cat.default([mul_tensor, permute_default, view_default]);  mul_tensor = permute_default = view_default = None
        view_default_3: "f32[3, 8192, 4, 49, 32]" = torch.ops.aten.view.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None
        permute_default_1: "f32[8192, 49, 3, 4, 32]" = torch.ops.aten.permute.default(view_default_3, [1, 3, 0, 2, 4]);  view_default_3 = None
        clone_default: "f32[8192, 49, 3, 4, 32]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_4: "f32[8192, 49, 384]" = torch.ops.aten.view.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        view_default_5: "f32[401408, 384]" = torch.ops.aten.view.default(view_default_4, _shape_param_5);  view_default_4 = _shape_param_5 = None
        permute_default_2: "f32[384, 401408]" = torch.ops.aten.permute.default(view_default_5, [1, 0])
        sum_dim_int_list: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_default_5, [0], True);  view_default_5 = None
        view_default_6: "f32[384]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_6);  sum_dim_int_list = _shape_param_6 = None
        return (permute_default_2, view_default_6)

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
