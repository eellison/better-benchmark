"""
Standalone repro captured via capture_hook.
Label: hf_BertForMaskedLM_train
Pattern hash: 2655cc8a0790
Shape hash: 8ae0f618
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
    def forward(self, arg0_1: "bf16[384, 512, 512]", arg1_1: "bf16[32, 12, 512, 512]", arg2_1: "i64[37]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        amax: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view, -inf);  view = None
        logical_not: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        where: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_1, arg1_1, convert_element_type_1);  logical_not_1 = arg1_1 = convert_element_type_1 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg2_1, 34);  arg2_1 = None
        inductor_random: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.1);  convert_element_type_2 = None
        mul: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt, where)
        mul_1: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        expand: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_1, _shape_param_2);  mul_1 = _shape_param_2 = None
        view_1: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand, _shape_param_3);  expand = _shape_param_3 = None
        permute: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_1, [0, 2, 1])
        return (where, gt, view_1, permute)



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
