"""
Standalone repro captured via capture_hook.
Label: vit_base_patch16_siglip_256
Pattern hash: fb3ad2fbbb4a
Shape hash: 09efb447
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
    def forward(self, arg0_1: "bf16[128, 12, 1, 64]", arg1_1: "bf16[768, 768]", arg2_1: "bf16[768]", arg3_1: "bf16[768]", arg4_1: "bf16[768]", arg5_1: "bf16[3072, 768]", arg6_1: "bf16[3072]", arg7_1: "bf16[768, 3072]", arg8_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        permute: "bf16[128, 1, 12, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[128, 1, 768]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        view_1: "bf16[128, 768]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[768, 768]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm: "bf16[128, 768]" = torch.ops.aten.addmm.default(arg2_1, view_1, permute_1);  arg2_1 = view_1 = permute_1 = None
        view_2: "bf16[128, 1, 768]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        convert_element_type: "f32[128, 1, 768]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 1, 1]" = var_mean[0]
        getitem_1: "f32[128, 1, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 1, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add: "f32[128, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[128, 1, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type_1: "bf16[128, 1, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view_3: "bf16[128, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        permute_2: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_1: "bf16[128, 3072]" = torch.ops.aten.addmm.default(arg6_1, view_3, permute_2);  arg6_1 = view_3 = permute_2 = None
        view_4: "bf16[128, 1, 3072]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        convert_element_type_2: "f32[128, 1, 3072]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_2: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5)
        mul_3: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476);  convert_element_type_2 = None
        erf: "f32[128, 1, 3072]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_2: "f32[128, 1, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_2, add_2);  mul_2 = add_2 = None
        convert_element_type_3: "bf16[128, 1, 3072]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        view_5: "bf16[128, 3072]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_5);  convert_element_type_3 = _shape_param_5 = None
        permute_3: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_2: "bf16[128, 768]" = torch.ops.aten.addmm.default(arg8_1, view_5, permute_3);  arg8_1 = view_5 = permute_3 = None
        view_6: "bf16[128, 1, 768]" = torch.ops.aten.view.default(addmm_2, _shape_param_6);  addmm_2 = _shape_param_6 = None
        add_3: "bf16[128, 1, 768]" = torch.ops.aten.add.Tensor(view_2, view_6);  view_2 = view_6 = None
        select: "bf16[128, 768]" = torch.ops.aten.select.int(add_3, 1, 0);  add_3 = None
        return select



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
