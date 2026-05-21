"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train_002
Pattern hash: 93cb2fd0355b
Shape hash: b2c6c352
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
_shapes_config = "(T([72, 512, 512], f32), T([24, 3, 256, 257], f32, stride=(525312, 131328, 513, 1)), T([24, 3, 256, 513], f32, stride=(525312, 131328, 513, 1)), T([24, 4, 256, 513], f32), T([2, 256, 12, 257], b8), T([2, 256, 12, 257], f32, stride=(789504, 257, 65792, 1)), T([2, 256, 12, 257], b8), T([2, 1024, 1, 513], f32), T([2, 1024, 1, 1], b8), T([], f32), T([36], i64), S([24, 3, 512, 1, 512]), S([24, 3, 512, 512]), S([24, 3, 512, 513]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([24, 4, -1]), S([24, 4, 256, 769]), S([96, 256, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_22: "f32[72, 512, 512]", slice_4: "f32[24, 3, 256, 257]", slice_3: "f32[24, 3, 256, 513]", full: "f32[24, 4, 256, 513]", convert_element_type: "b8[2, 256, 12, 257]", permute_12: "f32[2, 256, 12, 257]", convert_element_type_1: "b8[2, 256, 12, 257]", permute_25: "f32[2, 1024, 1, 513]", unsqueeze_11: "b8[2, 1024, 1, 1]", full_2: "f32[]", inductor_seeds: "i64[36]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        view_default: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_0);  bmm_22 = _shape_param_0 = None
        permute_default: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_default, [0, 1, 2, 4, 3]);  view_default = None
        view_default_1: "f32[24, 3, 512, 512]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        constant_pad_nd_default: "f32[24, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_default_1, [0, 0, 0, 1], 0.0);  view_default_1 = None
        view_default_2: "f32[24, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_default, _shape_param_2);  constant_pad_nd_default = _shape_param_2 = None
        slice_tensor: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_default_2, 2, 0, 256)
        slice_tensor_1: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 257);  slice_tensor = None
        copy_default: "f32[24, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_tensor_1);  slice_4 = slice_tensor_1 = None
        slice_scatter_default: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_default, 3, 256, 9223372036854775807);  slice_3 = copy_default = None
        slice_scatter_default_1: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_default, 1, 0, -1);  full = slice_scatter_default = None
        select_int: "f32[24, 512, 513]" = torch.ops.aten.select.int(view_default_2, 1, -1)
        slice_tensor_2: "f32[24, 256, 513]" = torch.ops.aten.slice.Tensor(select_int, 1, 256, 9223372036854775807);  select_int = None
        slice_tensor_3: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 2, 0, 257);  slice_tensor_2 = None
        select_int_1: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_1, 1, -1)
        slice_tensor_4: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_int_1, 2, 256, 9223372036854775807)
        copy_default_1: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_4, slice_tensor_3);  slice_tensor_4 = slice_tensor_3 = None
        slice_scatter_default_2: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_1, copy_default_1, 2, 256, 9223372036854775807);  select_int_1 = copy_default_1 = None
        select_scatter_default: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_1, slice_scatter_default_2, 1, -1);  slice_scatter_default_1 = slice_scatter_default_2 = None
        slice_tensor_5: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_default_2, 2, -257, -1)
        slice_tensor_6: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_5, 3, 257, 9223372036854775807);  slice_tensor_5 = None
        slice_tensor_7: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807)
        slice_tensor_8: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_7, 3, 0, 256)
        copy_default_2: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_tensor_8, slice_tensor_6);  slice_tensor_8 = slice_tensor_6 = None
        slice_scatter_default_3: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_7, copy_default_2, 3, 0, 256);  slice_tensor_7 = copy_default_2 = None
        slice_scatter_default_4: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default, slice_scatter_default_3, 1, 1, 9223372036854775807);  select_scatter_default = slice_scatter_default_3 = None
        select_int_2: "f32[24, 512, 513]" = torch.ops.aten.select.int(view_default_2, 1, 0);  view_default_2 = None
        slice_tensor_9: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_2, 1, 0, 255);  select_int_2 = None
        slice_tensor_10: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_9, 2, -255, 9223372036854775807);  slice_tensor_9 = None
        select_int_3: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_4, 1, 0)
        slice_tensor_11: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_3, 1, 1, 256)
        slice_tensor_12: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_11, 2, 1, 256)
        copy_default_3: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_tensor_12, slice_tensor_10);  slice_tensor_12 = slice_tensor_10 = None
        slice_scatter_default_5: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_11, copy_default_3, 2, 1, 256);  slice_tensor_11 = copy_default_3 = None
        slice_scatter_default_6: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_3, slice_scatter_default_5, 1, 1, 256);  select_int_3 = slice_scatter_default_5 = None
        select_scatter_default_1: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_4, slice_scatter_default_6, 1, 0);  slice_scatter_default_4 = slice_scatter_default_6 = None
        view_default_3: "f32[2, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_default_1, _shape_param_3);  select_scatter_default_1 = _shape_param_3 = None
        permute_default_1: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        slice_tensor_13: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_1, 1, 0, 256)
        slice_tensor_14: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_13, 3, 0, 257)
        where_self: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_12, slice_tensor_14);  convert_element_type = None
        copy_default_4: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_14, where_self);  slice_tensor_14 = where_self = None
        slice_scatter_default_7: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_13, copy_default_4, 3, 0, 257);  slice_tensor_13 = copy_default_4 = None
        slice_scatter_default_8: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_1, slice_scatter_default_7, 1, 0, 256);  permute_default_1 = slice_scatter_default_7 = None
        permute_default_2: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_8, [0, 2, 1, 3]);  slice_scatter_default_8 = None
        view_default_4: "f32[24, 4, 256, 513]" = torch.ops.aten.view.default(permute_default_2, _shape_param_4);  permute_default_2 = _shape_param_4 = None
        view_default_5: "f32[2, 12, 1024, 513]" = torch.ops.aten.view.default(view_default_4, _shape_param_5);  view_default_4 = _shape_param_5 = None
        permute_default_3: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        slice_tensor_15: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_3, 1, -256, 9223372036854775807)
        slice_tensor_16: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_15, 3, -257, 9223372036854775807)
        where_self_1: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_12, slice_tensor_16);  convert_element_type_1 = permute_12 = None
        copy_default_5: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_16, where_self_1);  slice_tensor_16 = where_self_1 = None
        slice_scatter_default_9: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_15, copy_default_5, 3, -257, 9223372036854775807);  slice_tensor_15 = copy_default_5 = None
        slice_scatter_default_10: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_3, slice_scatter_default_9, 1, -256, 9223372036854775807);  permute_default_3 = slice_scatter_default_9 = None
        permute_default_4: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_10, [0, 2, 1, 3]);  slice_scatter_default_10 = None
        permute_default_5: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_default_4, [0, 2, 1, 3]);  permute_default_4 = None
        add_tensor: "f32[2, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_default_5, permute_25);  permute_default_5 = permute_25 = None
        permute_default_6: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1, 3]);  add_tensor = None
        permute_default_7: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_default_6, [0, 2, 1, 3]);  permute_default_6 = None
        clone_default: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        amax_default: "f32[2, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_default, [-1], True)
        sub_tensor: "f32[2, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_default, amax_default);  clone_default = amax_default = None
        exp_default: "f32[2, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[2, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self_2: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_2, div_tensor);  unsqueeze_11 = full_2 = div_tensor = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 33);  inductor_seeds = None
        inductor_random_default: "f32[2, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([2, 1024, 12, 513], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[2, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_scalar, where_self_2);  gt_scalar = where_self_2 = None
        mul_tensor_1: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        permute_default_8: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_tensor_1, [0, 2, 1, 3]);  mul_tensor_1 = None
        clone_default_1: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        view_default_6: "f32[24, 4, 256, 513]" = torch.ops.aten.view.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        constant_pad_nd_default_1: "f32[24, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_default_6, [0, 257], 0.0);  view_default_6 = None
        view_default_7: "f32[24, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_default_1, _shape_param_7);  constant_pad_nd_default_1 = _shape_param_7 = None
        slice_tensor_17: "f32[24, 4, 196864]" = torch.ops.aten.slice.Tensor(view_default_7, 2, 0, -256);  view_default_7 = None
        view_default_8: "f32[24, 4, 256, 769]" = torch.ops.aten.view.default(slice_tensor_17, _shape_param_8);  slice_tensor_17 = _shape_param_8 = None
        slice_tensor_18: "f32[24, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_default_8, 3, 0, -1);  view_default_8 = None
        unsqueeze_default: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_18, 4);  slice_tensor_18 = None
        view_default_9: "f32[96, 256, 768]" = torch.ops.aten.view.default(unsqueeze_default, _shape_param_9);  unsqueeze_default = _shape_param_9 = None
        permute_default_9: "f32[96, 768, 256]" = torch.ops.aten.permute.default(view_default_9, [0, 2, 1]);  view_default_9 = None
        return permute_default_9



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
