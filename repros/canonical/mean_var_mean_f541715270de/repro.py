"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_infer
Pattern hash: f541715270de
Shape hash: e781f0b8
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
    def forward(self, arg0_1: "bf16[25216, 768]", arg1_1: "bf16[768]", arg2_1: "bf16[128, 197, 768]", arg3_1: "bf16[768]", arg4_1: "bf16[768]", _shape_param_0):
        # No stacktrace found for following nodes
        view: "bf16[128, 197, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "bf16[128, 197, 768]" = torch.ops.aten.mul.Tensor(arg1_1, view);  arg1_1 = view = None
        add: "bf16[128, 197, 768]" = torch.ops.aten.add.Tensor(arg2_1, mul);  arg2_1 = mul = None
        slice_1: "bf16[128, 196, 768]" = torch.ops.aten.slice.Tensor(add, 1, 1, 9223372036854775807);  add = None
        mean: "bf16[128, 768]" = torch.ops.aten.mean.dim(slice_1, [1]);  slice_1 = None
        convert_element_type: "f32[128, 768]" = torch.ops.prims.convert_element_type.default(mean, torch.float32);  mean = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [1], correction = 0, keepdim = True)
        getitem: "f32[128, 1]" = var_mean[0]
        getitem_1: "f32[128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul_1: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_2: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        convert_element_type_1: "bf16[128, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        return convert_element_type_1



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
