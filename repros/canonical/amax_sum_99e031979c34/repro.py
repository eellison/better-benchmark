"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 99e031979c34
Shape hash: b64f0e8a
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
    def forward(self, arg0_1: "bf16[288, 512, 512]", arg1_1: "bf16[96, 3, 256, 257]", arg2_1: "bf16[96, 3, 256, 513]", arg3_1: "bf16[96, 4, 256, 513]", arg4_1: "b8[8, 256, 12, 257]", arg5_1: "bf16[8, 256, 12, 257]", arg6_1: "b8[8, 256, 12, 257]", arg7_1: "bf16[8, 1024, 1, 513]", arg8_1: "b8[8, 1024, 1, 1]", arg9_1: "f32[]", arg10_1: "i64[36]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # No stacktrace found for following nodes
        view: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3]);  view = None
        view_1: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        constant_pad_nd: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1, [0, 0, 0, 1], 0.0);  view_1 = None
        view_2: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_2);  constant_pad_nd = _shape_param_2 = None
        slice_1: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_2, 2, 0, 256)
        slice_2: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 257);  slice_1 = None
        copy: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(arg1_1, slice_2);  arg1_1 = slice_2 = None
        slice_scatter: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(arg2_1, copy, 3, 256, 9223372036854775807);  arg2_1 = copy = None
        slice_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(arg3_1, slice_scatter, 1, 0, -1);  arg3_1 = slice_scatter = None
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
        where: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(arg4_1, arg5_1, slice_15);  arg4_1 = None
        copy_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_15, where);  slice_15 = where = None
        slice_scatter_7: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_14, copy_4, 3, 0, 257);  slice_14 = copy_4 = None
        slice_scatter_8: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1, slice_scatter_7, 1, 0, 256);  permute_1 = slice_scatter_7 = None
        permute_2: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_8, [0, 2, 1, 3]);  slice_scatter_8 = None
        view_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_2, _shape_param_4);  permute_2 = _shape_param_4 = None
        view_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_3: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        slice_16: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_3, 1, -256, 9223372036854775807)
        slice_17: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_16, 3, -257, 9223372036854775807)
        where_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(arg6_1, arg5_1, slice_17);  arg6_1 = arg5_1 = None
        copy_5: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_17, where_1);  slice_17 = where_1 = None
        slice_scatter_9: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_16, copy_5, 3, -257, 9223372036854775807);  slice_16 = copy_5 = None
        slice_scatter_10: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_3, slice_scatter_9, 1, -256, 9223372036854775807);  permute_3 = slice_scatter_9 = None
        permute_4: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_10, [0, 2, 1, 3]);  slice_scatter_10 = None
        permute_5: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_4, [0, 2, 1, 3]);  permute_4 = None
        add: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_5, arg7_1);  permute_5 = arg7_1 = None
        permute_6: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add, [0, 2, 1, 3]);  add = None
        permute_7: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1, 3]);  permute_6 = None
        convert_element_type: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_7, torch.float32)
        clone: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type, memory_format = torch.contiguous_format);  convert_element_type = None
        amax: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone, [-1], True)
        sub: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone, amax);  clone = None
        exp: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        where_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(arg8_1, arg9_1, div);  arg8_1 = arg9_1 = div = None
        convert_element_type_1: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_2, torch.bfloat16);  where_2 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg10_1, 9);  arg10_1 = None
        inductor_random: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_6, inductor_lookup_seed, 'rand');  _shape_param_6 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.1);  convert_element_type_2 = None
        mul: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt, convert_element_type_1);  convert_element_type_1 = None
        mul_1: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        permute_8: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_1, [0, 2, 1, 3]);  mul_1 = None
        clone_1: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_6: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        constant_pad_nd_1: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_6, _shape_param_8, 0.0);  view_6 = _shape_param_8 = None
        view_7: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_1, _shape_param_9);  constant_pad_nd_1 = _shape_param_9 = None
        slice_18: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_7, 2, 0, -256);  view_7 = None
        view_8: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_18, _shape_param_10);  slice_18 = _shape_param_10 = None
        slice_19: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_8, 3, 0, -1);  view_8 = None
        unsqueeze: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_19, 4);  slice_19 = None
        view_9: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze, _shape_param_11);  unsqueeze = _shape_param_11 = None
        permute_9: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_9, [0, 2, 1])
        return (permute_7, amax, sum_1, gt, view_9, permute_9)



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
