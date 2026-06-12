"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 89a9de6a6901
Shape hash: 55df3967
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[32, 128]", arg3_1: "bf16[32, 128, 250112]", arg4_1: "f32[4096, 1]", arg5_1: "f32[4096, 1]", arg6_1: "bf16[32, 128, 250112]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "i64[4096]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        unsqueeze: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[4096, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[4096, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[250112]" = torch.ops.prims.iota.default(250112, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 250112]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[4096, 250112]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[4096, 250112]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[4096, 250112]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[4096, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = None
        mul: "f32[4096, 250112]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "bf16[4096, 250112]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[4096, 250112]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        view_2: "bf16[4096, 250112]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[4096, 250112]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        sub_1: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        convert_element_type_3: "bf16[4096, 250112]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[4096, 250112]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp: "f32[4096, 250112]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_1: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[4096, 250112]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[4096, 250112]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_3: "bf16[32, 128, 250112]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        add: "bf16[32, 128, 250112]" = torch.ops.aten.add.Tensor(arg6_1, view_3);  arg6_1 = view_3 = None
        view_4: "bf16[4096, 250112]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        permute: "bf16[250112, 4096]" = torch.ops.aten.permute.default(view_4, [1, 0])
        return (full_1, view_4, permute)



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
