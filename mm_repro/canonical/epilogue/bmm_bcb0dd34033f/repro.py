"""Curated MM repro. Pattern hash: bcb0dd34033f."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[288, 512, 64]", arg1_1: "bf16[288, 64, 512]", arg2_1: "bf16[96, 3, 256, 257]", arg3_1: "bf16[96, 3, 256, 513]", arg4_1: "bf16[96, 4, 256, 513]", arg5_1: "b8[8, 256, 12, 257]", arg6_1: "bf16[8, 256, 12, 257]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3]);  view = None
        view_1: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        constant_pad_nd: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1, [0, 0, 0, 1], 0.0);  view_1 = None
        view_2: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_2);  constant_pad_nd = _shape_param_2 = None
        slice_1: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_2, 2, 0, 256)
        slice_2: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 257);  slice_1 = None
        copy: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(arg2_1, slice_2);  arg2_1 = slice_2 = None
        slice_scatter: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(arg3_1, copy, 3, 256, 9223372036854775807);  arg3_1 = copy = None
        slice_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(arg4_1, slice_scatter, 1, 0, -1);  arg4_1 = slice_scatter = None
        select: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_2, 1, -1)
        slice_3: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807);  select = None
        slice_4: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_3, 2, 0, 257);  slice_3 = None
        select_1: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        slice_5: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807)
        copy_1: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_5, slice_4);  slice_5 = slice_4 = None
        slice_scatter_2: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_1, copy_1, 2, 256, 9223372036854775807);  select_1 = copy_1 = None
        select_scatter: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_1, slice_scatter_2, 1, -1);  slice_scatter_1 = slice_scatter_2 = None
        slice_6: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_2, 2, -257, -1)
        slice_7: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_6, 3, 257, 9223372036854775807);  slice_6 = None
        slice_8: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_9: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 256)
        copy_2: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_9, slice_7);  slice_9 = slice_7 = None
        slice_scatter_3: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_8, copy_2, 3, 0, 256);  slice_8 = copy_2 = None
        slice_scatter_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_3, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_3 = None
        select_2: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_2, 1, 0);  view_2 = None
        slice_10: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_2, 1, 0, 255);  select_2 = None
        slice_11: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_10, 2, -255, 9223372036854775807);  slice_10 = None
        select_3: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_12: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_3, 1, 1, 256)
        slice_13: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_12, 2, 1, 256)
        copy_3: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_13, slice_11);  slice_13 = slice_11 = None
        slice_scatter_5: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_12, copy_3, 2, 1, 256);  slice_12 = copy_3 = None
        slice_scatter_6: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_3, slice_scatter_5, 1, 1, 256);  select_3 = slice_scatter_5 = None
        select_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_4, slice_scatter_6, 1, 0);  slice_scatter_4 = slice_scatter_6 = None
        view_3: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_1, _shape_param_3);  select_scatter_1 = _shape_param_3 = None
        permute_1: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        slice_14: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1, 1, 0, 256)
        slice_15: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 257)
        where: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(arg5_1, arg6_1, slice_15);  arg5_1 = arg6_1 = None
        copy_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_15, where);  slice_15 = where = None
        slice_scatter_7: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_14, copy_4, 3, 0, 257);  slice_14 = copy_4 = None
        slice_scatter_8: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1, slice_scatter_7, 1, 0, 256);  permute_1 = slice_scatter_7 = None
        permute_2: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_8, [0, 2, 1, 3]);  slice_scatter_8 = None
        permute_3: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_2, [0, 2, 1, 3]);  permute_2 = None
        return permute_3

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
