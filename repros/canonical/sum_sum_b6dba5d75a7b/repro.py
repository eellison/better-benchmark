"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: b6dba5d75a7b
Shape hash: 840e34c4
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
    def forward(self, arg0_1: "f32[16, 2]", arg1_1: "f32[16, 2]", _shape_param_0):
        # No stacktrace found for following nodes
        exp: "f32[16, 2]" = torch.ops.aten.exp.default(arg0_1);  arg0_1 = None
        sum_1: "f32[16, 1]" = torch.ops.aten.sum.dim_IntList(arg1_1, [-1], True)
        mul: "f32[16, 2]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub: "f32[16, 2]" = torch.ops.aten.sub.Tensor(arg1_1, mul);  arg1_1 = mul = None
        convert_element_type: "bf16[16, 2]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        permute: "bf16[2, 16]" = torch.ops.aten.permute.default(convert_element_type, [1, 0])
        sum_2: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0], True, dtype = torch.float32)
        view: "f32[2]" = torch.ops.aten.view.default(sum_2, _shape_param_0);  sum_2 = _shape_param_0 = None
        convert_element_type_1: "bf16[2]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_2: "f32[2]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        return (convert_element_type, permute, convert_element_type_2)



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
