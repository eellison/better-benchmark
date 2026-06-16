"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 5cb3d877d464
Shape hash: 66d9f76b
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
    def forward(self, arg0_1: "bf16[768, 196, 196]", arg1_1: "bf16[768, 200, 200]", arg2_1: "f32[128, 6, 196, 1]", arg3_1: "f32[128, 6, 196, 1]", arg4_1: "b8[128, 6, 196, 1]", arg5_1: "f32[128, 6, 196, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[128, 6, 196, 196]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[128, 6, 196, 196]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        slice_1: "bf16[768, 196, 200]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, -4);  arg1_1 = None
        slice_2: "bf16[768, 196, 196]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -4);  slice_1 = None
        view_1: "bf16[128, 6, 196, 196]" = torch.ops.aten.view.default(slice_2, _shape_param_1);  slice_2 = _shape_param_1 = None
        mul: "bf16[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_1, 0.125)
        convert_element_type_1: "f32[128, 6, 196, 196]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_2: "f32[128, 6, 196, 196]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_1: "f32[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        sub: "f32[128, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        mul_2: "f32[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub, 0.125);  sub = None
        sub_1: "f32[128, 6, 196, 196]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg3_1);  convert_element_type_1 = arg3_1 = None
        where: "f32[128, 6, 196, 196]" = torch.ops.aten.where.self(arg4_1, mul_2, sub_1);  arg4_1 = mul_2 = sub_1 = None
        exp: "f32[128, 6, 196, 196]" = torch.ops.aten.exp.default(where);  where = None
        div: "f32[128, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp, arg5_1);  exp = arg5_1 = None
        mul_3: "f32[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(convert_element_type, div);  convert_element_type = None
        sum_1: "f32[128, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [-1], True)
        neg: "f32[128, 6, 196, 196]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[128, 6, 196, 196]" = torch.ops.prims.fma.default(neg, sum_1, mul_3);  neg = sum_1 = mul_3 = None
        convert_element_type_3: "bf16[128, 6, 196, 196]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        mul_4: "bf16[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.125);  convert_element_type_3 = None
        view_2: "bf16[768, 196, 196]" = torch.ops.aten.view.default(mul_4, _shape_param_2);  mul_4 = _shape_param_2 = None
        return view_2



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
