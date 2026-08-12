"""
Standalone repro captured via capture_hook.
Label: timm_deit_tiny_patch16_224.fb_in1k_infer
Pattern hash: 813b85fdc01b
Shape hash: ad6d6241
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
    def forward(self, arg0_1: "bf16[25216, 192]", arg1_1: "bf16[128, 197, 192]", arg2_1: "bf16[192]", arg3_1: "bf16[192]", _shape_param_0):
        # No stacktrace found for following nodes
        view: "bf16[128, 197, 192]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "bf16[128, 197, 192]" = torch.ops.aten.add.Tensor(arg1_1, view);  arg1_1 = view = None
        convert_element_type: "f32[128, 197, 192]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 197, 1]" = var_mean[0]
        getitem_1: "f32[128, 197, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[128, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 197, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_2: "f32[128, 197, 192]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        convert_element_type_1: "bf16[128, 197, 192]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        select: "bf16[128, 192]" = torch.ops.aten.select.int(convert_element_type_1, 1, 0);  convert_element_type_1 = None
        clone: "bf16[128, 192]" = torch.ops.aten.clone.default(select);  select = None
        return clone



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
