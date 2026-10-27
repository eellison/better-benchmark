"""
Standalone repro captured via capture_hook.
Label: vit_base_patch16_siglip_256
Pattern hash: 1639565033e9
Shape hash: 59b1dfff
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
    def forward(self, arg0_1: "bf16[128, 768]", arg1_1: "bf16[768, 3072]", arg2_1: "bf16[128, 3072]", arg3_1: "bf16[128, 3072]", arg4_1: "bf16[3072, 768]", arg5_1: "bf16[128, 768]", arg6_1: "f32[768]", arg7_1: "bf16[128, 768]", arg8_1: "f32[128, 1, 1]", arg9_1: "f32[128, 1, 1]", arg10_1: "bf16[768, 768]", arg11_1: "bf16[128, 12, 1, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
        # No stacktrace found for following nodes
        full: "bf16[128, 1, 768]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        select_scatter: "bf16[128, 1, 768]" = torch.ops.aten.select_scatter.default(full, arg0_1, 1, 0);  full = arg0_1 = None
        view: "bf16[128, 768]" = torch.ops.aten.view.default(select_scatter, _shape_param_1);  _shape_param_1 = None
        mm: "bf16[128, 3072]" = torch.ops.aten.mm.default(view, arg1_1);  arg1_1 = None
        permute: "bf16[768, 128]" = torch.ops.aten.permute.default(view, [1, 0])
        mm_1: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute, arg2_1);  permute = arg2_1 = None
        sum_1: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32);  view = None
        view_1: "f32[768]" = torch.ops.aten.view.default(sum_1, _shape_param_2);  sum_1 = _shape_param_2 = None
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        view_2: "bf16[128, 1, 3072]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        convert_element_type_1: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_2: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        convert_element_type_3: "f32[128, 1, 3072]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        view_3: "bf16[128, 1, 3072]" = torch.ops.aten.view.default(arg3_1, _shape_param_4);  arg3_1 = _shape_param_4 = None
        convert_element_type_4: "f32[128, 1, 3072]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.7071067811865476)
        erf: "f32[128, 1, 3072]" = torch.ops.aten.erf.default(mul);  mul = None
        add: "f32[128, 1, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_1: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_2: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_4, convert_element_type_4)
        mul_3: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_2, -0.5);  mul_2 = None
        exp: "f32[128, 1, 3072]" = torch.ops.aten.exp.default(mul_3);  mul_3 = None
        mul_4: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_5: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_4, mul_4);  convert_element_type_4 = mul_4 = None
        add_1: "f32[128, 1, 3072]" = torch.ops.aten.add.Tensor(mul_1, mul_5);  mul_1 = mul_5 = None
        mul_6: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_3, add_1);  convert_element_type_3 = add_1 = None
        convert_element_type_5: "bf16[128, 1, 3072]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        view_4: "bf16[128, 3072]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_5);  convert_element_type_5 = _shape_param_5 = None
        mm_2: "bf16[128, 768]" = torch.ops.aten.mm.default(view_4, arg4_1);  arg4_1 = None
        permute_1: "bf16[3072, 128]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_3: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_1, arg5_1);  permute_1 = arg5_1 = None
        sum_2: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[3072]" = torch.ops.aten.view.default(sum_2, _shape_param_6);  sum_2 = _shape_param_6 = None
        convert_element_type_6: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[128, 1, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_7);  mm_2 = _shape_param_7 = None
        convert_element_type_7: "f32[128, 1, 768]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_8: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_9: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        mul_7: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, arg6_1);  arg6_1 = None
        mul_8: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(mul_7, 768)
        sum_3: "f32[128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_7, [2], True)
        view_7: "bf16[128, 1, 768]" = torch.ops.aten.view.default(arg7_1, _shape_param_8);  arg7_1 = _shape_param_8 = None
        convert_element_type_10: "f32[128, 1, 768]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        sub: "f32[128, 1, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_10, arg8_1);  convert_element_type_10 = arg8_1 = None
        mul_9: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(sub, arg9_1);  sub = None
        mul_10: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(mul_7, mul_9);  mul_7 = None
        sum_4: "f32[128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_10, [2], True);  mul_10 = None
        mul_11: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_4);  sum_4 = None
        sub_1: "f32[128, 1, 768]" = torch.ops.aten.sub.Tensor(mul_8, sum_3);  mul_8 = sum_3 = None
        sub_2: "f32[128, 1, 768]" = torch.ops.aten.sub.Tensor(sub_1, mul_11);  sub_1 = mul_11 = None
        div: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(arg9_1, 768);  arg9_1 = None
        mul_12: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_13: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, mul_9);  mul_9 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_13, [0, 1]);  mul_13 = None
        sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_7, [0, 1]);  convert_element_type_7 = None
        convert_element_type_11: "bf16[128, 1, 768]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        add_2: "bf16[128, 1, 768]" = torch.ops.aten.add.Tensor(select_scatter, convert_element_type_11);  select_scatter = convert_element_type_11 = None
        view_8: "bf16[128, 768]" = torch.ops.aten.view.default(add_2, _shape_param_9);  add_2 = _shape_param_9 = None
        mm_4: "bf16[128, 768]" = torch.ops.aten.mm.default(view_8, arg10_1);  arg10_1 = None
        permute_2: "bf16[768, 128]" = torch.ops.aten.permute.default(view_8, [1, 0])
        permute_3: "bf16[128, 1, 12, 64]" = torch.ops.aten.permute.default(arg11_1, [0, 2, 1, 3]);  arg11_1 = None
        view_9: "bf16[128, 1, 768]" = torch.ops.aten.view.default(permute_3, _shape_param_10);  permute_3 = _shape_param_10 = None
        view_10: "bf16[128, 768]" = torch.ops.aten.view.default(view_9, _shape_param_11);  view_9 = _shape_param_11 = None
        mm_5: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_2, view_10);  permute_2 = view_10 = None
        sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_11: "f32[768]" = torch.ops.aten.view.default(sum_7, _shape_param_12);  sum_7 = _shape_param_12 = None
        convert_element_type_12: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_11, torch.bfloat16);  view_11 = None
        view_12: "bf16[128, 1, 768]" = torch.ops.aten.view.default(mm_4, _shape_param_13);  mm_4 = _shape_param_13 = None
        convert_element_type_13: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_14: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_12, torch.float32);  convert_element_type_12 = None
        view_13: "bf16[128, 1, 12, 64]" = torch.ops.aten.view.default(view_12, _shape_param_14);  view_12 = _shape_param_14 = None
        permute_4: "bf16[128, 12, 1, 64]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        return (convert_element_type_1, convert_element_type_2, convert_element_type_8, convert_element_type_9, sum_5, sum_6, convert_element_type_13, convert_element_type_14, permute_4)



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
