"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/swin_base_patch4_window7_224/full_graph_001.py
Pattern hash: 3f4e9259510b
Shape hash: e07a7e88
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
    def forward(self, arg0_1: "bf16[401408, 384]", arg1_1: "bf16[384, 128]", arg2_1: "f32[128]", arg3_1: "bf16[128, 128, 56, 56]", arg4_1: "f32[128, 56, 56, 1]", arg5_1: "f32[128, 56, 56, 1]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[128, 56, 56, 1]", arg9_1: "f32[128, 56, 56, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[401408, 128]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[8192, 49, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "f32[8192, 7, 7, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        view_2: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        mul: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_3, arg2_1);  arg2_1 = None
        mul_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, 128)
        sum_1: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul, [3], True)
        permute_1: "bf16[128, 56, 56, 128]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 3, 1]);  arg3_1 = None
        convert_element_type_1: "f32[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        sub: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        mul_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = arg5_1 = None
        mul_3: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
        add: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_3, arg7_1);  mul_3 = arg7_1 = None
        sub_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(add, arg8_1);  add = arg8_1 = None
        mul_4: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_1, arg9_1);  sub_1 = None
        mul_5: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, mul_4);  mul = None
        sum_2: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [3], True);  mul_5 = None
        mul_6: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_4, sum_2);  sum_2 = None
        sub_2: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_3: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_2, mul_6);  sub_2 = mul_6 = None
        div: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(arg9_1, 128);  arg9_1 = None
        mul_7: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div, sub_3);  div = sub_3 = None
        mul_8: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_3, mul_4);  view_3 = mul_4 = None
        return (mul_7, mul_8)



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
