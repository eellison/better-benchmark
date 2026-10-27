"""
Standalone repro captured via capture_hook.
Label: vit_base_patch16_siglip_256
Pattern hash: f3c2b10a25e5
Shape hash: 28aa63eb
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
    def forward(self, arg0_1: "bf16[128, 768, 16, 16]", arg1_1: "f32[1, 256, 768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[2304]", arg5_1: "f32[2304, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[128, 768, 256]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[128, 256, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        add: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(permute, arg1_1);  permute = arg1_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 256, 1]" = var_mean[0]
        getitem_1: "f32[128, 256, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[128, 256, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 256, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1)
        mul: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_2: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        convert_element_type: "bf16[2304]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        convert_element_type_1: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_2: "bf16[128, 256, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_1: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_1);  convert_element_type_2 = _shape_param_1 = None
        permute_1: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[32768, 2304]" = torch.ops.aten.addmm.default(convert_element_type, view_1, permute_1);  convert_element_type = None
        view_2: "bf16[128, 256, 2304]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        view_3: "bf16[128, 256, 3, 12, 64]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        permute_2: "bf16[3, 128, 12, 256, 64]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
        unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
        getitem_2: "bf16[128, 12, 256, 64]" = unbind[0]
        getitem_3: "bf16[128, 12, 256, 64]" = unbind[1]
        getitem_4: "bf16[128, 12, 256, 64]" = unbind[2];  unbind = None
        permute_3: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (add, getitem_1, rsqrt, view_1, getitem_2, getitem_3, getitem_4, permute_3)



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
