"""
Standalone repro captured via capture_hook.
Label: hf_XGLMForCausalLM_train
Pattern hash: 68fcffe5c7fb
Shape hash: 8da16745
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
    def forward(self, arg0_1: "bf16[512, 128, 128]", arg1_1: "b8[512, 128, 128]", arg2_1: "bf16[512, 128, 128]", arg3_1: "f32[32, 1, 128, 128]", arg4_1: "f32[512, 128, 1]", arg5_1: "f32[512, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        convert_element_type: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        add: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view, arg3_1);  view = arg3_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum: "f32[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add, full)
        view_1: "f32[512, 128, 128]" = torch.ops.aten.view.default(maximum, _shape_param_1);  maximum = _shape_param_1 = None
        sub: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_1, arg4_1);  view_1 = arg4_1 = None
        exp: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp, arg5_1);  exp = arg5_1 = None
        mul_2: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(mul_1, div);  mul_1 = None
        sum_1: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [-1], True)
        neg: "f32[512, 128, 128]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[512, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul_2);  neg = sum_1 = mul_2 = None
        view_2: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(fma, _shape_param_2);  fma = _shape_param_2 = None
        div_1: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Scalar(view_2, 2)
        eq: "b8[32, 16, 128, 128]" = torch.ops.aten.eq.Tensor(add, full)
        where: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(eq, div_1, view_2);  eq = div_1 = view_2 = None
        lt: "b8[32, 16, 128, 128]" = torch.ops.aten.lt.Tensor(add, full);  add = full = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(lt, full_1, where);  lt = full_1 = where = None
        convert_element_type_2: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        view_3: "bf16[512, 128, 128]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_3);  convert_element_type_2 = _shape_param_3 = None
        return view_3



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
