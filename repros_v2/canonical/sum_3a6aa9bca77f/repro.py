"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 3a6aa9bca77f
Shape hash: 57ead6f2
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
    def forward(self, arg0_1: "bf16[131072, 288]", arg1_1: "bf16[131072, 288]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[512, 256, 288]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[512, 256, 288]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[512, 256, 288]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        convert_element_type_1: "f32[512, 256, 288]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        sigmoid: "f32[512, 256, 288]" = torch.ops.aten.sigmoid.default(convert_element_type_1)
        mul: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = None
        sub: "f32[512, 256, 288]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_1: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub);  convert_element_type_1 = sub = None
        add: "f32[512, 256, 288]" = torch.ops.aten.add.Tensor(mul_1, 1);  mul_1 = None
        mul_2: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_2: "bf16[512, 256, 288]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        view_2: "bf16[131072, 288]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        permute: "bf16[288, 131072]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_1: "f32[1, 288]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[288]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type_3: "bf16[288]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_4: "f32[288]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (view_2, permute, convert_element_type_4)



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
