"""
Standalone repro captured via capture_hook.
Label: repros/models/torchbench/train/nvidia_deeprecommender/full_graph_000.py
Pattern hash: 2fe2aeba588e
Shape hash: 57fbbea2
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
    def forward(self, arg0_1: "f32[512]", arg1_1: "f32[512, 197951]", arg2_1: "f32[256, 197951]"):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[512, 197951]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_2: "bf16[256, 197951]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        permute: "bf16[197951, 512]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[256, 512]" = torch.ops.aten.addmm.default(convert_element_type, convert_element_type_2, permute);  convert_element_type = convert_element_type_2 = permute = None
        convert_element_type_3: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        gt: "b8[256, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_3, 0)
        mul: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.0507009873554805)
        mul_1: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.0);  convert_element_type_3 = None
        expm1: "f32[256, 512]" = torch.ops.aten.expm1.default(mul_1);  mul_1 = None
        mul_2: "f32[256, 512]" = torch.ops.aten.mul.Tensor(expm1, 1.7580993408473766);  expm1 = None
        where: "f32[256, 512]" = torch.ops.aten.where.self(gt, mul, mul_2);  gt = mul = mul_2 = None
        convert_element_type_4: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        return convert_element_type_4



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
