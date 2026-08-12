"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 3d664201719d
Shape hash: 4205ff34
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
    def forward(self, arg0_1: "bf16[2048, 768]", arg1_1: "i64[61]", arg2_1: "f32[16, 128, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[16, 128, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg1_1, 22);  arg1_1 = None
        inductor_random: "f32[16, 128, 768]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[16, 128, 768]" = torch.ops.aten.gt.Scalar(convert_element_type, 0.1);  convert_element_type = None
        mul: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(gt, view);  view = None
        mul_1: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(arg2_1, mul_1);  arg2_1 = mul_1 = None
        mean: "f32[16, 128, 1]" = torch.ops.aten.mean.dim(add, [-1], True)
        var: "f32[16, 128, 1]" = torch.ops.aten.var.correction(add, [-1], correction = 1.0, keepdim = True)
        sqrt: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var);  var = None
        sub: "f32[16, 128, 768]" = torch.ops.aten.sub.Tensor(add, mean);  mean = None
        mul_2: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg3_1, sub);  arg3_1 = None
        add_1: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(sqrt, 1e-06)
        div: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_2, add_1);  mul_2 = add_1 = None
        add_2: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(div, arg4_1);  div = arg4_1 = None
        convert_element_type_1: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_1: "bf16[2048, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        return (gt, add, sqrt, sub, view_1)



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
