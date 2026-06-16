"""
Standalone repro captured via capture_hook.
Label: genai_CrossEntropyBackward_static
Pattern hash: b2873b4c9052
Shape hash: ed4a60c9
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
    def forward(self, arg0_1: "bf16[8192, 262144]", arg1_1: "f32[8192, 1]", arg2_1: "f32[8192, 1]", arg3_1: "i64[8192]", arg4_1: "bf16[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        sub: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type, arg1_1);  convert_element_type = arg1_1 = None
        sub_1: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(sub, arg2_1);  sub = arg2_1 = None
        convert_element_type_1: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_2: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        exp: "f32[8192, 262144]" = torch.ops.aten.exp.default(convert_element_type_2);  convert_element_type_2 = None
        unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, 1);  arg3_1 = None
        ne: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        expand: "i64[8192, 262144]" = torch.ops.aten.expand.default(where, _shape_param_0);  where = _shape_param_0 = None
        iota: "i64[262144]" = torch.ops.prims.iota.default(262144, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view: "i64[1, 262144]" = torch.ops.aten.view.default(iota, _shape_param_1);  iota = _shape_param_1 = None
        eq: "b8[8192, 262144]" = torch.ops.aten.eq.Tensor(expand, view);  expand = view = None
        full_1: "f32[]" = torch.ops.aten.full.default([], -1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8192, 262144]" = torch.ops.aten.where.self(eq, full_1, full_2);  eq = full_1 = full_2 = None
        expand_1: "bf16[8192]" = torch.ops.aten.expand.default(arg4_1, _shape_param_2);  arg4_1 = _shape_param_2 = None
        unsqueeze_1: "bf16[8192, 1]" = torch.ops.aten.unsqueeze.default(expand_1, 1);  expand_1 = None
        full_3: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[8192, 1]" = torch.ops.aten.where.self(ne, unsqueeze_1, full_3);  ne = unsqueeze_1 = full_3 = None
        mul: "f32[8192, 262144]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        mul_1: "f32[8192, 262144]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        convert_element_type_3: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        return convert_element_type_3



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
