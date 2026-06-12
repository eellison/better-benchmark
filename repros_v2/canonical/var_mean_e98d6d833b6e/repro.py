"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: e98d6d833b6e
Shape hash: 3162d0ee
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
    def forward(self, arg0_1: "bf16[128, 640, 7, 7]", arg1_1: "f32[640]", arg2_1: "f32[640]"):
        # No stacktrace found for following nodes
        permute: "bf16[128, 7, 7, 640]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 3, 1]);  arg0_1 = None
        convert_element_type: "f32[128, 7, 7, 640]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 7, 7, 1]" = var_mean[0]
        getitem_1: "f32[128, 7, 7, 1]" = var_mean[1];  var_mean = None
        add: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 7, 7, 640]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = None
        mul: "f32[128, 7, 7, 640]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 7, 7, 640]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1: "f32[128, 7, 7, 640]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        permute_1: "f32[128, 640, 7, 7]" = torch.ops.aten.permute.default(add_1, [0, 3, 1, 2]);  add_1 = None
        convert_element_type_1: "bf16[128, 640, 7, 7]" = torch.ops.prims.convert_element_type.default(permute_1, torch.bfloat16);  permute_1 = None
        return (getitem_1, rsqrt, convert_element_type_1)



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
