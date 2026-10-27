"""
Standalone repro captured via capture_hook.
Label: AllenaiLongformerBase
Pattern hash: bb9a4b50ddd5
Shape hash: 70720859
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
    def forward(self, arg0_1: "bf16[9437184]", arg1_1: "bf16[6291456]", arg2_1: "bf16[8192, 768]", arg3_1: "bf16[768, 768]", arg4_1: "bf16[6291456]", arg5_1: "bf16[768, 768]", arg6_1: "bf16[768, 768]", arg7_1: "f32[8, 1024, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23):
        # No stacktrace found for following nodes
        as_strided: "bf16[96, 1536, 64]" = torch.ops.aten.as_strided.default(arg0_1, _shape_param_0, _shape_param_1, 0);  arg0_1 = _shape_param_0 = _shape_param_1 = None
        constant_pad_nd: "bf16[96, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided, _shape_param_2);  as_strided = _shape_param_2 = None
        view: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_3);  constant_pad_nd = _shape_param_3 = None
        permute: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        permute_1: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute, [1, 0, 2, 3]);  permute = None
        clone: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_1: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        as_strided_1: "bf16[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(arg1_1, _shape_param_5, _shape_param_6, 0);  arg1_1 = _shape_param_5 = _shape_param_6 = None
        view_2: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(as_strided_1, _shape_param_7);  as_strided_1 = _shape_param_7 = None
        view_3: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_2, _shape_param_8);  view_2 = _shape_param_8 = None
        permute_2: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        permute_3: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_2, [1, 0, 2, 3]);  permute_2 = None
        view_4: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(permute_3, _shape_param_9);  permute_3 = _shape_param_9 = None
        div: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(view_4, 8.0);  view_4 = None
        sum_1: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1, [0, 1], True, dtype = torch.float32)
        view_5: "f32[768]" = torch.ops.aten.view.default(sum_1, _shape_param_10);  sum_1 = _shape_param_10 = None
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[8192, 768]" = torch.ops.aten.view.default(view_1, _shape_param_11);  view_1 = _shape_param_11 = None
        permute_4: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_6, [1, 0])
        mm: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_4, arg2_1);  permute_4 = None
        mm_1: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_6, arg3_1);  view_6 = arg3_1 = None
        view_7: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_12);  mm_1 = _shape_param_12 = None
        convert_element_type_1: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        convert_element_type_2: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_3: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        as_strided_2: "bf16[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(arg4_1, _shape_param_13, _shape_param_14, 0);  arg4_1 = _shape_param_13 = _shape_param_14 = None
        view_8: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(as_strided_2, _shape_param_15);  as_strided_2 = _shape_param_15 = None
        view_9: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_8, _shape_param_16);  view_8 = _shape_param_16 = None
        permute_5: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        permute_6: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_5, [1, 0, 2, 3]);  permute_5 = None
        view_10: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(permute_6, _shape_param_17);  permute_6 = _shape_param_17 = None
        sum_2: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_10, [0, 1], True, dtype = torch.float32)
        view_11: "f32[768]" = torch.ops.aten.view.default(sum_2, _shape_param_18);  sum_2 = _shape_param_18 = None
        convert_element_type_4: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_11, torch.bfloat16);  view_11 = None
        view_12: "bf16[8192, 768]" = torch.ops.aten.view.default(view_10, _shape_param_19);  view_10 = _shape_param_19 = None
        permute_7: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_12, [1, 0])
        mm_2: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_7, arg2_1);  permute_7 = None
        mm_3: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_12, arg5_1);  view_12 = arg5_1 = None
        view_13: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_3, _shape_param_20);  mm_3 = _shape_param_20 = None
        convert_element_type_5: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        add: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_5);  convert_element_type_1 = convert_element_type_5 = None
        convert_element_type_6: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_7: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        sum_3: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div, [0, 1], True, dtype = torch.float32)
        view_14: "f32[768]" = torch.ops.aten.view.default(sum_3, _shape_param_21);  sum_3 = _shape_param_21 = None
        convert_element_type_8: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_14, torch.bfloat16);  view_14 = None
        view_15: "bf16[8192, 768]" = torch.ops.aten.view.default(div, _shape_param_22);  div = _shape_param_22 = None
        permute_8: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_15, [1, 0])
        mm_4: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_8, arg2_1);  permute_8 = arg2_1 = None
        mm_5: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_15, arg6_1);  view_15 = arg6_1 = None
        view_16: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_5, _shape_param_23);  mm_5 = _shape_param_23 = None
        convert_element_type_9: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view_16, torch.float32);  view_16 = None
        add_1: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(add, convert_element_type_9);  add = convert_element_type_9 = None
        convert_element_type_10: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_11: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        permute_9: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(add_1, [1, 0, 2]);  add_1 = None
        add_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(arg7_1, permute_9);  arg7_1 = permute_9 = None
        return (convert_element_type_2, convert_element_type_3, convert_element_type_6, convert_element_type_7, convert_element_type_10, convert_element_type_11, add_2)



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
