"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_infer
Pattern hash: af50781fc699
Shape hash: 99deda8b
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
    def forward(self, arg0_1: "bf16[256, 512, 512]", arg1_1: "bf16[256, 512, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3]);  view = None
        view_1: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        view_2: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(arg1_1, _shape_param_2);  arg1_1 = _shape_param_2 = None
        permute_1: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_2, [0, 1, 2, 4, 3]);  view_2 = None
        view_3: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_1, _shape_param_3);  permute_1 = _shape_param_3 = None
        view_4: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_3, _shape_param_4);  view_3 = _shape_param_4 = None
        slice_1: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_4, 2, 1, 9223372036854775807);  view_4 = None
        view_5: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_1, _shape_param_5);  slice_1 = _shape_param_5 = None
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_5, [None, None, None, iota]);  view_5 = iota = None
        add: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_1, index);  view_1 = index = None
        add_1: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add, 0);  add = None
        mul: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_1, 0.125)
        convert_element_type: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        eq: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type, convert_element_type)
        abs_1: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type)
        ne: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_1, inf);  abs_1 = None
        mul_1: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq, ne);  eq = ne = None
        logical_not: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_1);  mul_1 = None
        any_1: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not, [3], True);  logical_not = None
        logical_not_1: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        convert_element_type_1: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        mul_2: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        amax: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_2, [3], True)
        sub: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_2, amax);  mul_2 = amax = None
        mul_3: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub, 0.125);  sub = None
        amax_1: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type, [3], True)
        sub_1: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type, amax_1);  convert_element_type = amax_1 = None
        where: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_1, mul_3, sub_1);  logical_not_1 = mul_3 = sub_1 = None
        exp: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where);  where = None
        sum_1: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_2: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        unsqueeze: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 4);  convert_element_type_2 = None
        view_6: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze, _shape_param_6);  unsqueeze = _shape_param_6 = None
        return view_6



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
