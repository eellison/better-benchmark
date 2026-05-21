"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: 27183a793fcd
Shape hash: b6754764
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 30524], f32), T([32768, 384], f32), T([256, 128], i64, gen=Index(30522)), T([], f32), S([256, 128, 384]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[512, 30524]", mm_724: "f32[32768, 384]", arg0_1: "i64[256, 128]", full_1: "f32[]", _shape_param_0):
        # No stacktrace found for following nodes
        slice_tensor: "f32[512, 30522]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -2);  mm = None
        slice_tensor_1: "f32[128, 30522]" = torch.ops.aten.slice.Tensor(slice_tensor, 0, 0, 128)
        slice_tensor_2: "f32[384, 30522]" = torch.ops.aten.slice.Tensor(slice_tensor, 0, 128, 512);  slice_tensor = None
        permute_default: "f32[30522, 128]" = torch.ops.aten.permute.default(slice_tensor_1, [1, 0]);  slice_tensor_1 = None
        view_default: "f32[256, 128, 384]" = torch.ops.aten.view.default(mm_724, _shape_param_0);  mm_724 = _shape_param_0 = None
        slice_tensor_3: "f32[256, 128, 128]" = torch.ops.aten.slice.Tensor(view_default, 2, 0, 128)
        slice_tensor_4: "f32[256, 128, 128]" = torch.ops.aten.slice.Tensor(view_default, 2, 128, 256)
        slice_tensor_5: "f32[256, 128, 128]" = torch.ops.aten.slice.Tensor(view_default, 2, 256, 384);  view_default = None
        constant_pad_nd_default: "f32[256, 127, 128]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_5, [0, 0, -1, 0, 0, 0]);  slice_tensor_5 = None
        full_default: "f32[256, 128, 128]" = torch.ops.aten.full.default([256, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[256, 128, 128]" = torch.ops.aten.slice_scatter.default(full_default, constant_pad_nd_default, 1, 0, -1);  constant_pad_nd_default = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(slice_tensor_4, slice_scatter_default);  slice_tensor_4 = slice_scatter_default = None
        constant_pad_nd_default_1: "f32[256, 127, 128]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_3, [0, 0, 0, -1, 0, 0]);  slice_tensor_3 = None
        slice_scatter_default_1: "f32[256, 128, 128]" = torch.ops.aten.slice_scatter.default(full_default, constant_pad_nd_default_1, 1, 1, 9223372036854775807);  full_default = constant_pad_nd_default_1 = None
        add_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor, slice_scatter_default_1);  add_tensor = slice_scatter_default_1 = None
        eq_scalar: "b8[256, 128]" = torch.ops.aten.eq.Scalar(arg0_1, 0)
        unsqueeze_default: "b8[256, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[256, 128, 128]" = torch.ops.aten.where.self(unsqueeze_default, full_1, add_tensor_1);  unsqueeze_default = full_1 = add_tensor_1 = None
        full_default_1: "f32[30522, 128]" = torch.ops.aten.full.default([30522, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[30522, 128]" = torch.ops.aten.index_put.default(full_default_1, [arg0_1], where_self, True);  full_default_1 = arg0_1 = where_self = None
        add_tensor_2: "f32[30522, 128]" = torch.ops.aten.add.Tensor(permute_default, index_put_default);  permute_default = index_put_default = None
        return (slice_tensor_2, add_tensor_2)



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
