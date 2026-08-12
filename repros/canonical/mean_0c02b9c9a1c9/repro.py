"""
Standalone repro captured via capture_hook.
Label: torchbench_phlippe_densenet_infer
Pattern hash: 0c02b9c9a1c9
Shape hash: 2c3fa82b
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
    def forward(self, arg0_1: "bf16[128, 16, 4, 4]", arg1_1: "bf16[128, 168, 4, 4]", arg2_1: "bf16[184]", arg3_1: "bf16[184]", arg4_1: "bf16[184]", arg5_1: "bf16[184]", _shape_param_0):
        # No stacktrace found for following nodes
        cat: "bf16[128, 184, 4, 4]" = torch.ops.aten.cat.default([arg0_1, arg1_1], 1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[184]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        unsqueeze: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        unsqueeze_1: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[128, 184, 4, 4]" = torch.ops.aten.sub.Tensor(cat, unsqueeze_1);  cat = unsqueeze_1 = None
        convert_element_type_1: "f32[184]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        add: "f32[184]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
        sqrt: "f32[184]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[128, 184, 4, 4]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "bf16[184, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_5: "bf16[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[128, 184, 4, 4]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "bf16[184, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_7: "bf16[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[128, 184, 4, 4]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_2: "bf16[128, 184, 4, 4]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        relu: "bf16[128, 184, 4, 4]" = torch.ops.aten.relu.default(convert_element_type_2);  convert_element_type_2 = None
        mean: "bf16[128, 184, 1, 1]" = torch.ops.aten.mean.dim(relu, [-1, -2], True);  relu = None
        view: "bf16[128, 184]" = torch.ops.aten.view.default(mean, _shape_param_0);  mean = _shape_param_0 = None
        return view



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
