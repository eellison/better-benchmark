"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 444779f98932
Shape hash: 47e7063f
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
    def forward(self, arg0_1: "bf16[384, 256, 768]", arg1_1: "bf16[96, 4, 256, 769]", arg2_1: "bf16[96, 4, 197120]", arg3_1: "b8[8, 1024, 12, 513]", arg4_1: "b8[8, 1024, 1, 1]", arg5_1: "f32[]", arg6_1: "bf16[8, 1024, 12, 513]", arg7_1: "f32[8, 1024, 12, 1]", arg8_1: "f32[8, 1024, 12, 1]", arg9_1: "bf16[8, 256, 12, 257]", arg10_1: "b8[8, 256, 12, 257]", arg11_1: "bf16[]", arg12_1: "bf16[8, 256, 12, 513]", arg13_1: "bf16[8, 1024, 12, 513]", arg14_1: "b8[8, 256, 12, 257]", arg15_1: "bf16[96, 255, 255]", arg16_1: "bf16[96, 255, 513]", arg17_1: "bf16[96, 512, 513]", arg18_1: "bf16[96, 3, 512, 513]", arg19_1: "bf16[96, 3, 256, 256]", arg20_1: "bf16[96, 3, 256, 513]", arg21_1: "bf16[96, 256, 257]", arg22_1: "bf16[96, 256, 513]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16):
        # No stacktrace found for following nodes
        view: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        squeeze: "bf16[96, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view, 4);  view = None
        slice_scatter: "bf16[96, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(arg1_1, squeeze, 3, 0, -1);  arg1_1 = squeeze = None
        view_1: "bf16[96, 4, 196864]" = torch.ops.aten.view.default(slice_scatter, _shape_param_1);  slice_scatter = _shape_param_1 = None
        slice_scatter_1: "bf16[96, 4, 197120]" = torch.ops.aten.slice_scatter.default(arg2_1, view_1, 2, 0, -256);  arg2_1 = view_1 = None
        view_2: "bf16[96, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_1, _shape_param_2);  slice_scatter_1 = _shape_param_2 = None
        constant_pad_nd: "bf16[96, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        view_3: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_4);  constant_pad_nd = _shape_param_4 = None
        permute: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        convert_element_type: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        mul: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute, mul);  permute = mul = None
        clone: "bf16[8, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_1, memory_format = torch.contiguous_format);  mul_1 = None
        convert_element_type_1: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(clone, torch.float32);  clone = None
        where: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(arg4_1, arg5_1, convert_element_type_1);  arg4_1 = arg5_1 = convert_element_type_1 = None
        convert_element_type_2: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        clone_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_2, memory_format = torch.contiguous_format);  convert_element_type_2 = None
        sub: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_1, arg7_1);  clone_1 = arg7_1 = None
        exp: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, arg8_1);  exp = arg8_1 = None
        mul_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where, div);  where = None
        sum_1: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [-1], True)
        neg: "f32[8, 1024, 12, 513]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[8, 1024, 12, 513]" = torch.ops.prims.fma.default(neg, sum_1, mul_2);  neg = sum_1 = mul_2 = None
        convert_element_type_3: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        permute_1: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_3, [0, 2, 1, 3]);  convert_element_type_3 = None
        clone_2: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_2, _shape_param_5);  clone_2 = _shape_param_5 = None
        view_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_4, _shape_param_6);  view_4 = _shape_param_6 = None
        permute_2: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone_3: "bf16[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format)
        copy: "bf16[8, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_2, clone_3);  permute_2 = clone_3 = None
        permute_3: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(copy, [0, 2, 1, 3]);  copy = None
        view_6: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_3, _shape_param_7);  permute_3 = _shape_param_7 = None
        view_7: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_6, _shape_param_8);  view_6 = _shape_param_8 = None
        permute_4: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        slice_1: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_4, 1, -256, 9223372036854775807)
        slice_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1, 3, -257, 9223372036854775807)
        clone_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format)
        copy_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2, arg9_1);  slice_2 = None
        slice_scatter_2: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1, copy_1, 3, -257, 9223372036854775807);  slice_1 = copy_1 = None
        slice_scatter_3: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_4, slice_scatter_2, 1, -256, 9223372036854775807);  permute_4 = slice_scatter_2 = None
        permute_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_3, [0, 2, 1, 3]);  slice_scatter_3 = None
        view_8: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_5, _shape_param_9);  permute_5 = _shape_param_9 = None
        where_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(arg10_1, arg11_1, clone_4);  arg10_1 = clone_4 = None
        slice_scatter_4: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(arg12_1, where_1, 3, -257, 9223372036854775807);  where_1 = None
        slice_scatter_5: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(arg13_1, slice_scatter_4, 1, -256, 9223372036854775807);  slice_scatter_4 = None
        permute_6: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_5, [0, 2, 1, 3]);  slice_scatter_5 = None
        clone_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_9: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_5, _shape_param_10);  clone_5 = _shape_param_10 = None
        add: "bf16[96, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_8, view_9);  view_8 = view_9 = None
        view_10: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(add, _shape_param_11);  add = _shape_param_11 = None
        permute_7: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        slice_3: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_7, 1, 0, 256)
        slice_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 257)
        clone_6: "bf16[8, 256, 12, 257]" = torch.ops.aten.clone.default(slice_4, memory_format = torch.contiguous_format)
        copy_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_4, arg9_1);  slice_4 = arg9_1 = None
        slice_scatter_6: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_2, 3, 0, 257);  slice_3 = copy_2 = None
        slice_scatter_7: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_7, slice_scatter_6, 1, 0, 256);  permute_7 = slice_scatter_6 = None
        permute_8: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_7, [0, 2, 1, 3]);  slice_scatter_7 = None
        view_11: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_8, _shape_param_12);  permute_8 = _shape_param_12 = None
        where_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(arg14_1, arg11_1, clone_6);  arg14_1 = arg11_1 = clone_6 = None
        slice_scatter_8: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(arg12_1, where_2, 3, 0, 257);  arg12_1 = where_2 = None
        slice_scatter_9: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(arg13_1, slice_scatter_8, 1, 0, 256);  arg13_1 = slice_scatter_8 = None
        permute_9: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_9, [0, 2, 1, 3]);  slice_scatter_9 = None
        clone_7: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_12: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_7, _shape_param_13);  clone_7 = _shape_param_13 = None
        add_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_11, view_12);  view_11 = view_12 = None
        select: "bf16[96, 256, 513]" = torch.ops.aten.select.int(add_1, 1, 0)
        slice_5: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select, 1, 1, 256)
        slice_6: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_5, 2, 1, 256)
        clone_8: "bf16[96, 255, 255]" = torch.ops.aten.clone.default(slice_6, memory_format = torch.contiguous_format)
        copy_3: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_6, arg15_1);  slice_6 = arg15_1 = None
        slice_scatter_10: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_5, copy_3, 2, 1, 256);  slice_5 = copy_3 = None
        slice_scatter_11: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select, slice_scatter_10, 1, 1, 256);  select = slice_scatter_10 = None
        select_scatter: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_1, slice_scatter_11, 1, 0);  add_1 = slice_scatter_11 = None
        slice_scatter_12: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(arg16_1, clone_8, 2, -255, 9223372036854775807);  arg16_1 = clone_8 = None
        slice_scatter_13: "bf16[96, 512, 513]" = torch.ops.aten.slice_scatter.default(arg17_1, slice_scatter_12, 1, 0, 255);  slice_scatter_12 = None
        select_scatter_1: "bf16[96, 3, 512, 513]" = torch.ops.aten.select_scatter.default(arg18_1, slice_scatter_13, 1, 0);  slice_scatter_13 = None
        slice_7: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_8: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 256)
        clone_9: "bf16[96, 3, 256, 256]" = torch.ops.aten.clone.default(slice_8, memory_format = torch.contiguous_format)
        copy_4: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_8, arg19_1);  slice_8 = arg19_1 = None
        slice_scatter_14: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_4, 3, 0, 256);  slice_7 = copy_4 = None
        slice_scatter_15: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_14, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_14 = None
        slice_scatter_16: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(arg20_1, clone_9, 3, 257, 9223372036854775807);  clone_9 = None
        slice_scatter_17: "bf16[96, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(arg18_1, slice_scatter_16, 2, -257, -1);  slice_scatter_16 = None
        add_2: "bf16[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_1, slice_scatter_17);  select_scatter_1 = slice_scatter_17 = None
        select_1: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, -1)
        slice_9: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807)
        clone_10: "bf16[96, 256, 257]" = torch.ops.aten.clone.default(slice_9, memory_format = torch.contiguous_format)
        copy_5: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_9, arg21_1);  slice_9 = arg21_1 = None
        slice_scatter_18: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_1, copy_5, 2, 256, 9223372036854775807);  select_1 = copy_5 = None
        select_scatter_2: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_15, slice_scatter_18, 1, -1);  slice_scatter_15 = slice_scatter_18 = None
        slice_scatter_19: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(arg22_1, clone_10, 2, 0, 257);  arg22_1 = clone_10 = None
        slice_scatter_20: "bf16[96, 512, 513]" = torch.ops.aten.slice_scatter.default(arg17_1, slice_scatter_19, 1, 256, 9223372036854775807);  arg17_1 = slice_scatter_19 = None
        select_scatter_3: "bf16[96, 3, 512, 513]" = torch.ops.aten.select_scatter.default(arg18_1, slice_scatter_20, 1, -1);  slice_scatter_20 = None
        add_3: "bf16[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_2, select_scatter_3);  add_2 = select_scatter_3 = None
        slice_10: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 0, -1);  select_scatter_2 = None
        slice_11: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_10, 3, 256, 9223372036854775807);  slice_10 = None
        clone_11: "bf16[96, 3, 256, 257]" = torch.ops.aten.clone.default(slice_11, memory_format = torch.contiguous_format);  slice_11 = None
        slice_scatter_21: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(arg20_1, clone_11, 3, 0, 257);  arg20_1 = clone_11 = None
        slice_scatter_22: "bf16[96, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(arg18_1, slice_scatter_21, 2, 0, 256);  arg18_1 = slice_scatter_21 = None
        add_4: "bf16[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_3, slice_scatter_22);  add_3 = slice_scatter_22 = None
        view_13: "bf16[96, 3, 513, 512]" = torch.ops.aten.view.default(add_4, _shape_param_14);  add_4 = _shape_param_14 = None
        constant_pad_nd_1: "bf16[96, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_13, [0, 0, 0, -1]);  view_13 = None
        view_14: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_1, _shape_param_15);  constant_pad_nd_1 = _shape_param_15 = None
        permute_10: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_14, [0, 1, 2, 4, 3]);  view_14 = None
        view_15: "bf16[288, 512, 512]" = torch.ops.aten.view.default(permute_10, _shape_param_16);  permute_10 = _shape_param_16 = None
        return view_15



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
