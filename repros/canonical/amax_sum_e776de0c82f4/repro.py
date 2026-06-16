"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: e776de0c82f4
Shape hash: 1715052e
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
    def forward(self, arg0_1: "bf16[192, 128, 128]", arg1_1: "i64[84]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_1: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        amax: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_1, [-1], True)
        sub: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_1, amax);  convert_element_type_1 = None
        exp: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_2: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg1_1, 49);  arg1_1 = None
        inductor_random: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type_3: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_3, 0.1);  convert_element_type_3 = None
        mul: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt, convert_element_type_2);  convert_element_type_2 = None
        mul_1: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        expand: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_1, _shape_param_2);  mul_1 = _shape_param_2 = None
        view_1: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand, _shape_param_3);  expand = _shape_param_3 = None
        permute: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1])
        return (amax, sum_1, gt, view_1, permute)



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
