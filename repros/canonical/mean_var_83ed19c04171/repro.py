"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_infer
Pattern hash: 83ed19c04171
Shape hash: 1c404995
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
    def forward(self, arg0_1: "bf16[2048, 768]", arg1_1: "bf16[16, 128, 768]", arg2_1: "bf16[768]", arg3_1: "bf16[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[16, 128, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(arg1_1, view);  arg1_1 = view = None
        mean: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add, [-1], True)
        sub: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add, mean);  mean = None
        mul: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg2_1, sub);  arg2_1 = sub = None
        convert_element_type: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type, [-1], correction = 1.0, keepdim = True);  convert_element_type = None
        sqrt: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var);  var = None
        convert_element_type_1: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt, torch.bfloat16);  sqrt = None
        add_1: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-06);  convert_element_type_1 = None
        div: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul, add_1);  mul = add_1 = None
        add_2: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div, arg3_1);  div = arg3_1 = None
        view_1: "bf16[2048, 768]" = torch.ops.aten.view.default(add_2, _shape_param_1);  add_2 = _shape_param_1 = None
        return (add, view_1)



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
