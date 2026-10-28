"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/AllenaiLongformerBase/full_graph_002.py
Pattern hash: 1c2c06e56a5b
Shape hash: 2a8bfb39
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[288, 512, 64]", arg1_1: "bf16[288, 64, 512]", arg2_1: "b8[8, 256, 12, 257]", arg3_1: "b8[8, 256, 12, 257]", arg4_1: "bf16[8, 1024, 1, 513]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # No stacktrace found for following nodes
        full: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        slice_1: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_2: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_3: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2, 3, 256, 9223372036854775807);  slice_2 = None
        bmm: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        permute: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3]);  view = None
        view_1: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute, _shape_param_2);  permute = _shape_param_2 = None
        constant_pad_nd: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1, [0, 0, 0, 1], 0.0);  view_1 = None
        view_2: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_3);  constant_pad_nd = _shape_param_3 = None
        slice_4: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_2, 2, 0, 256)
        slice_5: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 257);  slice_4 = None
        copy: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_3, slice_5);  slice_3 = slice_5 = None
        slice_scatter: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1, copy, 3, 256, 9223372036854775807);  slice_1 = copy = None
        slice_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter, 1, 0, -1);  full = slice_scatter = None
        select: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        select_1: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        slice_6: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807);  select_1 = None
        select_2: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_2, 1, -1)
        slice_7: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_2, 1, 256, 9223372036854775807);  select_2 = None
        slice_8: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_7, 2, 0, 257);  slice_7 = None
        copy_1: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_6, slice_8);  slice_6 = slice_8 = None
        slice_scatter_2: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select, copy_1, 2, 256, 9223372036854775807);  select = copy_1 = None
        select_scatter: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_1, slice_scatter_2, 1, -1);  slice_scatter_1 = slice_scatter_2 = None
        slice_9: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_10: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_11: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_10, 3, 0, 256);  slice_10 = None
        slice_12: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_2, 2, -257, -1)
        slice_13: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_12, 3, 257, 9223372036854775807);  slice_12 = None
        copy_2: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_11, slice_13);  slice_11 = slice_13 = None
        slice_scatter_3: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_9, copy_2, 3, 0, 256);  slice_9 = copy_2 = None
        slice_scatter_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_3, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_3 = None
        select_3: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_14: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_3, 1, 1, 256)
        select_4: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_15: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_4, 1, 1, 256);  select_4 = None
        slice_16: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_15, 2, 1, 256);  slice_15 = None
        select_5: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_2, 1, 0);  view_2 = None
        slice_17: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_5, 1, 0, 255);  select_5 = None
        slice_18: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_17, 2, -255, 9223372036854775807);  slice_17 = None
        copy_3: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_16, slice_18);  slice_16 = slice_18 = None
        slice_scatter_5: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_14, copy_3, 2, 1, 256);  slice_14 = copy_3 = None
        slice_scatter_6: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_3, slice_scatter_5, 1, 1, 256);  select_3 = slice_scatter_5 = None
        select_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_4, slice_scatter_6, 1, 0);  slice_scatter_4 = slice_scatter_6 = None
        view_3: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_1, _shape_param_4);  _shape_param_4 = None
        permute_1: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        slice_19: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1, 1, 0, 256)
        view_4: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_1, _shape_param_5);  _shape_param_5 = None
        permute_2: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        slice_20: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_2, 1, 0, 256);  permute_2 = None
        slice_21: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 257);  slice_20 = None
        full_1: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_6, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        permute_3: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_1, [0, 2, 1, 3]);  full_1 = None
        view_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_1, _shape_param_7);  select_scatter_1 = _shape_param_7 = None
        permute_4: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        slice_22: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_4, 1, 0, 256);  permute_4 = None
        slice_23: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 257);  slice_22 = None
        where: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(arg2_1, permute_3, slice_23);  arg2_1 = permute_3 = slice_23 = None
        copy_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_21, where);  slice_21 = where = None
        slice_scatter_7: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_19, copy_4, 3, 0, 257);  slice_19 = copy_4 = None
        slice_scatter_8: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1, slice_scatter_7, 1, 0, 256);  permute_1 = slice_scatter_7 = None
        permute_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_8, [0, 2, 1, 3]);  slice_scatter_8 = None
        view_6: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_5, _shape_param_8);  permute_5 = _shape_param_8 = None
        view_7: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_6, _shape_param_9);  _shape_param_9 = None
        permute_6: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        slice_24: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_6, 1, -256, 9223372036854775807)
        view_8: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_6, _shape_param_10);  _shape_param_10 = None
        permute_7: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        slice_25: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_7, 1, -256, 9223372036854775807);  permute_7 = None
        slice_26: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_25, 3, -257, 9223372036854775807);  slice_25 = None
        full_2: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_11, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_11 = None
        permute_8: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_2, [0, 2, 1, 3]);  full_2 = None
        view_9: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_6, _shape_param_12);  view_6 = _shape_param_12 = None
        permute_9: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        slice_27: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_9, 1, -256, 9223372036854775807);  permute_9 = None
        slice_28: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_27, 3, -257, 9223372036854775807);  slice_27 = None
        where_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(arg3_1, permute_8, slice_28);  arg3_1 = permute_8 = slice_28 = None
        copy_5: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_26, where_1);  slice_26 = where_1 = None
        slice_scatter_9: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_24, copy_5, 3, -257, 9223372036854775807);  slice_24 = copy_5 = None
        slice_scatter_10: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_6, slice_scatter_9, 1, -256, 9223372036854775807);  permute_6 = slice_scatter_9 = None
        permute_10: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_10, [0, 2, 1, 3]);  slice_scatter_10 = None
        permute_11: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_10, [0, 2, 1, 3]);  permute_10 = None
        add: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_11, arg4_1);  permute_11 = arg4_1 = None
        permute_12: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add, [0, 2, 1, 3]);  add = None
        permute_13: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_12, [0, 2, 1, 3]);  permute_12 = None
        convert_element_type: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_13, torch.float32);  permute_13 = None
        clone: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type, memory_format = torch.contiguous_format);  convert_element_type = None
        return clone



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
