"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/XGLMForCausalLM/full_graph_006.py
Pattern hash: bcd7d76d653b
Shape hash: d40dc1d8
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
    def forward(self, arg0_1: "bf16[512, 128, 64]", arg1_1: "bf16[512, 64, 128]", arg2_1: "b8[512, 128, 128]", arg3_1: "bf16[512, 128, 128]", arg4_1: "f32[32, 1, 128, 128]", arg5_1: "f32[512, 128, 1]", arg6_1: "f32[512, 128, 1]", arg7_1: "bf16[512, 64, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
        convert_element_type_1: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        add: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view, arg4_1);  view = arg4_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum: "f32[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add, full)
        view_1: "f32[512, 128, 128]" = torch.ops.aten.view.default(maximum, _shape_param_1);  maximum = _shape_param_1 = None
        sub: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_1, arg5_1);  view_1 = arg5_1 = None
        exp: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp, arg6_1);  exp = arg6_1 = None
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
        bmm_1: "bf16[512, 64, 128]" = torch.ops.aten.bmm.default(arg7_1, view_3);  arg7_1 = None
        permute: "bf16[512, 128, 64]" = torch.ops.aten.permute.default(bmm_1, [0, 2, 1]);  bmm_1 = None
        view_4: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(permute, _shape_param_4);  permute = _shape_param_4 = None
        permute_1: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        view_5: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(permute_1, _shape_param_5);  permute_1 = _shape_param_5 = None
        clone: "bf16[32, 128, 1024]" = torch.ops.aten.clone.default(view_5, memory_format = torch.contiguous_format);  view_5 = None
        view_6: "bf16[4096, 1024]" = torch.ops.aten.view.default(clone, _shape_param_6);  clone = _shape_param_6 = None
        permute_2: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_6, [1, 0])
        return (view_3, view_6, permute_2)



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
