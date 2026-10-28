"""
Standalone repro captured via capture_hook.
Label: repros/models/torchbench/infer/mobilenet_v3_large/full_graph_000.py
Pattern hash: 564ae4a21730
Shape hash: 15ded4b2
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
    def forward(self, arg0_1: "bf16[960]", arg1_1: "bf16[32, 960, 7, 7]", arg2_1: "bf16[960]", arg3_1: "bf16[960]", arg4_1: "bf16[960]", arg5_1: "bf16[1280, 960]", arg6_1: "bf16[1280]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[960]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        unsqueeze: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        unsqueeze_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[32, 960, 7, 7]" = torch.ops.aten.sub.Tensor(arg1_1, unsqueeze_1);  arg1_1 = unsqueeze_1 = None
        convert_element_type_1: "f32[960]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0.001);  convert_element_type_1 = None
        sqrt: "f32[960]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "bf16[960, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_5: "bf16[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "bf16[960, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_7: "bf16[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[32, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_2: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        convert_element_type_3: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        add_2: "f32[32, 960, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_3, 3)
        clamp_min: "f32[32, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_2, 0);  add_2 = None
        clamp_max: "f32[32, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        mul_3: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_3, clamp_max);  convert_element_type_3 = clamp_max = None
        div: "f32[32, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_3, 6);  mul_3 = None
        convert_element_type_4: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mean: "bf16[32, 960, 1, 1]" = torch.ops.aten.mean.dim(convert_element_type_4, [-1, -2], True);  convert_element_type_4 = None
        view: "bf16[32, 960]" = torch.ops.aten.view.default(mean, _shape_param_0);  mean = _shape_param_0 = None
        permute: "bf16[960, 1280]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "bf16[32, 1280]" = torch.ops.aten.addmm.default(arg6_1, view, permute);  arg6_1 = view = permute = None
        convert_element_type_5: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        add_3: "f32[32, 1280]" = torch.ops.aten.add.Tensor(convert_element_type_5, 3)
        clamp_min_1: "f32[32, 1280]" = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
        clamp_max_1: "f32[32, 1280]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
        mul_4: "f32[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_5, clamp_max_1);  convert_element_type_5 = clamp_max_1 = None
        div_1: "f32[32, 1280]" = torch.ops.aten.div.Tensor(mul_4, 6);  mul_4 = None
        convert_element_type_6: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        return convert_element_type_6



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
