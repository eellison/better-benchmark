"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/MobileBertForMaskedLM/full_graph_001.py
Pattern hash: bdd66b8c1580
Shape hash: d5deab4c
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
    def forward(self, arg0_1: "bf16[1024, 128, 32]", arg1_1: "bf16[1024, 32, 128]", arg2_1: "b8[256, 4, 128, 128]", arg3_1: "bf16[256, 4, 128, 128]", arg4_1: "bf16[1024, 32, 128]", arg5_1: "bf16[128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        bmm: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mul: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(view, mul);  view = mul = None
        convert_element_type_1: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        convert_element_type_2: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        mul_2: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_2);  convert_element_type_1 = None
        sum_1: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [-1], True)
        neg: "f32[256, 4, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_2);  convert_element_type_2 = None
        fma: "f32[256, 4, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul_2);  neg = sum_1 = mul_2 = None
        convert_element_type_3: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_1: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_1);  convert_element_type_3 = _shape_param_1 = None
        bmm_1: "bf16[1024, 32, 128]" = torch.ops.aten.bmm.default(arg4_1, view_1);  arg4_1 = None
        view_2: "bf16[256, 4, 32, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_2);  bmm_1 = _shape_param_2 = None
        mul_3: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(view_2, 0.4204482076268573);  view_2 = None
        permute: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(mul_3, [0, 1, 3, 2]);  mul_3 = None
        permute_1: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(permute, [0, 2, 1, 3]);  permute = None
        view_3: "bf16[256, 128, 128]" = torch.ops.aten.view.default(permute_1, _shape_param_3);  permute_1 = _shape_param_3 = None
        clone: "bf16[256, 128, 128]" = torch.ops.aten.clone.default(view_3, memory_format = torch.contiguous_format);  view_3 = None
        view_4: "bf16[32768, 128]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        mm: "bf16[32768, 128]" = torch.ops.aten.mm.default(view_4, arg5_1);  arg5_1 = None
        permute_2: "bf16[128, 32768]" = torch.ops.aten.permute.default(view_4, [1, 0])
        view_5: "bf16[256, 128, 128]" = torch.ops.aten.view.default(mm, _shape_param_5);  mm = _shape_param_5 = None
        convert_element_type_4: "f32[256, 128, 128]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        return (view_1, view_4, permute_2, convert_element_type_4)



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
