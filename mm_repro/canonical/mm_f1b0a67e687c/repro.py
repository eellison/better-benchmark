"""
Standalone repro captured via capture_hook.
Label: repros/models/torchbench/train/BERT_pytorch/full_graph_001.py
Pattern hash: f1b0a67e687c
Shape hash: b850c895
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
    def forward(self, arg0_1: "bf16[2048, 3072]", arg1_1: "bf16[3072, 768]", arg2_1: "f32[768]", arg3_1: "f32[16, 128, 768]", arg4_1: "f32[16, 128, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[2048, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg2_1, arg3_1)
        add: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg4_1, 1e-06);  arg4_1 = None
        div: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul, add);  mul = None
        div_1: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div, add);  div = None
        neg: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type)
        mul_1: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg, div_1);  neg = div_1 = None
        div_2: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type, add);  convert_element_type = add = None
        mul_2: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_2, arg2_1);  arg2_1 = None
        mul_3: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_2, arg3_1);  div_2 = arg3_1 = None
        neg_1: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_2)
        return (mul_1, mul_2, mul_3, neg_1)



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
