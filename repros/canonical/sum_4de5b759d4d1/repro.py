"""
Standalone repro captured via capture_hook.
Label: torchbench_dlrm_train_001
Pattern hash: 4de5b759d4d1
Shape hash: 216a2c8f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 100], f32), T([2048, 64, 9], f32), T([2048, 9, 64], f32), T([2048, 64], b8), T([], f32), S([2048, 576]), S([64]))"

class Repro(torch.nn.Module):
    def forward(self, mm_5: "f32[2048, 100]", bmm: "f32[2048, 64, 9]", bmm_1: "f32[2048, 9, 64]", arg53_1: "b8[2048, 64]", full: "f32[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        slice_tensor: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(mm_5, 1, 0, 64);  mm_5 = None
        permute_default: "f32[2048, 9, 64]" = torch.ops.aten.permute.default(bmm, [0, 2, 1]);  bmm = None
        add_tensor: "f32[2048, 9, 64]" = torch.ops.aten.add.Tensor(bmm_1, permute_default);  bmm_1 = permute_default = None
        view_default: "f32[2048, 576]" = torch.ops.aten.view.default(add_tensor, _shape_param_0);  add_tensor = _shape_param_0 = None
        slice_tensor_1: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_default, 1, 0, 64)
        slice_tensor_2: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_default, 1, 64, 128)
        slice_tensor_3: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_default, 1, 128, 192)
        slice_tensor_4: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_default, 1, 192, 256)
        slice_tensor_5: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_default, 1, 256, 320)
        slice_tensor_6: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_default, 1, 320, 384)
        slice_tensor_7: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_default, 1, 384, 448)
        slice_tensor_8: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_default, 1, 448, 512)
        slice_tensor_9: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_default, 1, 512, 576);  view_default = None
        add_tensor_1: "f32[2048, 64]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        where_self: "f32[2048, 64]" = torch.ops.aten.where.self(arg53_1, full, add_tensor_1);  arg53_1 = full = add_tensor_1 = None
        permute_default_1: "f32[64, 2048]" = torch.ops.aten.permute.default(where_self, [1, 0])
        sum_dim_int_list: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(where_self, [0], True);  where_self = None
        view_default_1: "f32[64]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        return (slice_tensor_2, slice_tensor_3, slice_tensor_4, slice_tensor_5, slice_tensor_6, slice_tensor_7, slice_tensor_8, slice_tensor_9, permute_default_1, view_default_1)

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
