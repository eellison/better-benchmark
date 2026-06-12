"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: d582b23f1a3f
Shape hash: 74bd5ffe
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
    def forward(self, arg0_1: "bf16[8192, 240]", arg1_1: "bf16[512, 16, 240]", arg2_1: "f32[240]", arg3_1: "f32[240]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[512, 16, 240]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "bf16[512, 16, 240]" = torch.ops.aten.add.Tensor(arg1_1, view);  arg1_1 = view = None
        convert_element_type: "f32[512, 16, 240]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1]" = var_mean[0]
        getitem_1: "f32[512, 16, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = None
        mul: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_2: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        view_1: "f32[128, 4, 16, 240]" = torch.ops.aten.view.default(add_2, _shape_param_1);  add_2 = _shape_param_1 = None
        permute: "f32[128, 240, 16, 4]" = torch.ops.aten.permute.default(view_1, [0, 3, 2, 1]);  view_1 = None
        clone: "f32[128, 240, 16, 4]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "f32[122880, 4, 2, 2]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        permute_1: "f32[122880, 2, 4, 2]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone_1: "f32[122880, 2, 4, 2]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "f32[128, 240, 8, 8]" = torch.ops.aten.view.default(clone_1, _shape_param_3);  clone_1 = _shape_param_3 = None
        convert_element_type_1: "bf16[128, 240, 8, 8]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        return (add, getitem_1, rsqrt, convert_element_type_1)



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
