"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/MobileBertForMaskedLM/full_graph_000.py
Pattern hash: fe4891df3e8d
Shape hash: 37923383
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
    def forward(self, arg0_1: "bf16[128, 128]", arg1_1: "bf16[128]", arg2_1: "bf16[32768, 128]", arg3_1: "bf16[128, 128]", arg4_1: "bf16[128]", arg5_1: "bf16[32768, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # No stacktrace found for following nodes
        permute: "bf16[128, 128]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None
        view: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_1: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        mul: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_1, 0.4204482076268573);  permute_1 = None
        expand: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul, _shape_param_2);  mul = _shape_param_2 = None
        clone: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        permute_2: "bf16[128, 128]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_1: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg4_1, arg5_1, permute_2);  arg4_1 = arg5_1 = permute_2 = None
        view_3: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        view_4: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_3, _shape_param_5);  view_3 = _shape_param_5 = None
        permute_3: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_4: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_1: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_4, 0.4204482076268573);  permute_4 = None
        expand_1: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_1, _shape_param_6);  mul_1 = _shape_param_6 = None
        clone_1: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_5: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        bmm: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_2, view_5);  view_2 = view_5 = None
        view_6: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_8);  bmm = _shape_param_8 = None
        eq: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_6, -inf)
        logical_not: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_9, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_9 = None
        convert_element_type: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        amax: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_1);  logical_not_1 = full = convert_element_type_1 = None
        expand_2: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where, _shape_param_10);  where = _shape_param_10 = None
        view_7: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_2, _shape_param_11);  expand_2 = _shape_param_11 = None
        return view_7



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
