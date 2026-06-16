"""
Standalone repro captured via capture_hook.
Label: torchbench_tts_angular_train
Pattern hash: 20ba1158c4b0
Shape hash: ee4b9eab
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
    def forward(self, arg0_1: "f32[64, 256]", arg1_1: "f32[64, 1]", arg2_1: "bf16[64, 256]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        neg: "f32[64, 256]" = torch.ops.aten.neg.default(arg0_1)
        clamp_min: "f32[64, 1]" = torch.ops.aten.clamp_min.default(arg1_1, 1e-12)
        expand: "f32[64, 256]" = torch.ops.aten.expand.default(clamp_min, _shape_param_0);  clamp_min = _shape_param_0 = None
        div: "f32[64, 256]" = torch.ops.aten.div.Tensor(arg2_1, expand)
        div_1: "f32[64, 256]" = torch.ops.aten.div.Tensor(div, expand);  div = None
        mul: "f32[64, 256]" = torch.ops.aten.mul.Tensor(neg, div_1);  neg = div_1 = None
        div_2: "f32[64, 256]" = torch.ops.aten.div.Tensor(arg0_1, expand);  arg0_1 = expand = None
        convert_element_type: "bf16[64, 256]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        sum_1: "f32[64, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True, dtype = torch.float32);  mul = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ge: "b8[64, 1]" = torch.ops.aten.ge.Scalar(arg1_1, 1e-12)
        where: "f32[64, 1]" = torch.ops.aten.where.self(ge, sum_1, full);  ge = sum_1 = None
        div_3: "f32[64, 256]" = torch.ops.aten.div.Tensor(arg2_1, arg1_1);  arg2_1 = None
        eq: "b8[64, 1]" = torch.ops.aten.eq.Scalar(arg1_1, 0);  arg1_1 = None
        where_1: "f32[64, 256]" = torch.ops.aten.where.self(eq, full, div_3);  eq = full = div_3 = None
        mul_1: "f32[64, 256]" = torch.ops.aten.mul.Tensor(where, where_1);  where = where_1 = None
        convert_element_type_1: "bf16[64, 256]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        add: "bf16[64, 256]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        full_1: "bf16[64, 50, 256]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        select_scatter: "bf16[64, 50, 256]" = torch.ops.aten.select_scatter.default(full_1, add, 1, -1);  full_1 = add = None
        return select_scatter



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
