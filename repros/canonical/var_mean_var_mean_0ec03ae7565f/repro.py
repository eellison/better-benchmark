"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer
Pattern hash: 0ec03ae7565f
Shape hash: b68c1040
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
    def forward(self, arg0_1: "bf16[128, 128, 56, 56]", arg1_1: "bf16[128]", arg2_1: "bf16[128]", arg3_1: "bf16[128]", arg4_1: "bf16[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        permute: "bf16[128, 56, 56, 128]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 3, 1]);  arg0_1 = None
        convert_element_type: "f32[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 56, 56, 1]" = var_mean[0]
        getitem_1: "f32[128, 56, 56, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        convert_element_type_1: "bf16[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        convert_element_type_2: "f32[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_2, [3], correction = 0, keepdim = True)
        getitem_2: "f32[128, 56, 56, 1]" = var_mean_1[0]
        getitem_3: "f32[128, 56, 56, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_3);  convert_element_type_2 = getitem_3 = None
        add_2: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_2, arg3_1);  mul_2 = arg3_1 = None
        add_3: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_3, arg4_1);  mul_3 = arg4_1 = None
        convert_element_type_3: "bf16[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_0);  convert_element_type_3 = _shape_param_0 = None
        permute_1: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2, 4, 5]);  view = None
        clone: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_1: "bf16[8192, 7, 7, 128]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        view_3: "bf16[401408, 128]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        return (convert_element_type_1, view_3)



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
