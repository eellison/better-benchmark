"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_train
Pattern hash: 1bdd2568d0b8
Shape hash: 0fd7b2d6
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "b8[8, 2]", arg3_1: "b8[8, 1]", arg4_1: "bf16[8, 2]", arg5_1: "bf16[8, 2]", arg6_1: "i64[8]", arg7_1: "i64[8]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where: "f32[8, 2]" = torch.ops.aten.where.self(arg2_1, scalar_tensor_1, scalar_tensor);  arg2_1 = scalar_tensor_1 = scalar_tensor = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8, 1]" = torch.ops.aten.where.self(arg3_1, div, full);  arg3_1 = div = full = None
        mul: "f32[8, 2]" = torch.ops.aten.mul.Tensor(where, where_1);  where = where_1 = None
        convert_element_type: "bf16[8, 2]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[8, 2]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        convert_element_type_2: "f32[8, 2]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        amax: "f32[8, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [1], True)
        sub: "f32[8, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[8, 2]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[8, 2]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_3: "bf16[8, 2]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[8, 2]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp_1: "f32[8, 2]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_2: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[8, 2]" = torch.ops.aten.mul.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        sub_2: "f32[8, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[8, 2]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        add: "bf16[8, 2]" = torch.ops.aten.add.Tensor(arg5_1, convert_element_type_5);  arg5_1 = convert_element_type_5 = None
        full_1: "bf16[8, 1024, 2]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        index_put: "bf16[8, 1024, 2]" = torch.ops.aten.index_put.default(full_1, [arg6_1, arg7_1], add, True);  full_1 = arg6_1 = arg7_1 = add = None
        view: "bf16[8192, 2]" = torch.ops.aten.view.default(index_put, _shape_param_1);  index_put = _shape_param_1 = None
        permute: "bf16[2, 8192]" = torch.ops.aten.permute.default(view, [1, 0])
        return (view, permute)



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
