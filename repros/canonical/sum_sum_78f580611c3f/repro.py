"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 78f580611c3f
Shape hash: 9cebd270
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
    def forward(self, arg0_1: "bf16[256, 128, 1, 1]", arg1_1: "f32[256, 128, 1, 1]", arg2_1: "f32[1, 256, 1]", arg3_1: "f32[256]", arg4_1: "f32[256, 1, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        convert_element_type: "f32[256, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "f32[1, 256, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        sum_1: "f32[256]" = torch.ops.aten.sum.dim_IntList(view, [0, 2])
        view_1: "f32[1, 256, 128]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        sub: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        mul: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(view, sub)
        sum_2: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul, [0, 2]);  mul = None
        mul_1: "f32[256]" = torch.ops.aten.mul.Tensor(sum_1, 0.0078125);  sum_1 = None
        unsqueeze: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1, 0);  mul_1 = None
        unsqueeze_1: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        mul_2: "f32[256]" = torch.ops.aten.mul.Tensor(sum_2, 0.0078125)
        mul_3: "f32[256]" = torch.ops.aten.mul.Tensor(arg3_1, arg3_1)
        mul_4: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        unsqueeze_2: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_4, 0);  mul_4 = None
        unsqueeze_3: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
        mul_5: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg4_1, 0.08838834764831845);  arg4_1 = None
        view_2: "f32[256]" = torch.ops.aten.view.default(mul_5, [-1]);  mul_5 = None
        mul_6: "f32[256]" = torch.ops.aten.mul.Tensor(arg3_1, view_2);  view_2 = None
        unsqueeze_4: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_5: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        mul_7: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        sub_1: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view, mul_7);  view = mul_7 = None
        sub_2: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_1);  sub_1 = unsqueeze_1 = None
        mul_8: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_5);  sub_2 = unsqueeze_5 = None
        mul_9: "f32[256]" = torch.ops.aten.mul.Tensor(sum_2, arg3_1);  sum_2 = arg3_1 = None
        view_3: "f32[256, 1, 1, 1]" = torch.ops.aten.view.default(mul_9, _shape_param_2);  mul_9 = _shape_param_2 = None
        mul_10: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_3, 0.08838834764831845);  view_3 = None
        view_4: "f32[256, 128, 1, 1]" = torch.ops.aten.view.default(mul_8, _shape_param_3);  mul_8 = _shape_param_3 = None
        return (mul_10, view_4)



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
