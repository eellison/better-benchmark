"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: 7c4a1a0a4d8d
Shape hash: fd237320
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
    def forward(self, arg0_1: "bf16[128, 640]", arg1_1: "f32[640]", arg2_1: "bf16[128, 1, 1, 640]", arg3_1: "f32[128, 1, 1, 1]", arg4_1: "f32[128, 1, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 640]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "f32[128, 640, 1, 1]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        permute: "f32[128, 1, 1, 640]" = torch.ops.aten.permute.default(view, [0, 2, 3, 1]);  view = None
        mul: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(permute, arg1_1);  arg1_1 = None
        mul_1: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(mul, 640)
        sum_1: "f32[128, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul, [3], True)
        convert_element_type_1: "f32[128, 1, 1, 640]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub: "f32[128, 1, 1, 640]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg3_1);  convert_element_type_1 = arg3_1 = None
        mul_2: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = None
        mul_3: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_2: "f32[128, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [3], True);  mul_3 = None
        mul_4: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_1: "f32[128, 1, 1, 640]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[128, 1, 1, 640]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(arg4_1, 640);  arg4_1 = None
        mul_5: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(permute, mul_2);  mul_2 = None
        sum_3: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1, 2]);  mul_6 = None
        sum_4: "f32[640]" = torch.ops.aten.sum.dim_IntList(permute, [0, 1, 2]);  permute = None
        convert_element_type_2: "bf16[128, 1, 1, 640]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        permute_1: "bf16[128, 640, 1, 1]" = torch.ops.aten.permute.default(convert_element_type_2, [0, 3, 1, 2]);  convert_element_type_2 = None
        squeeze: "bf16[128, 640, 1]" = torch.ops.aten.squeeze.dim(permute_1, 3);  permute_1 = None
        squeeze_1: "bf16[128, 640]" = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
        full: "bf16[81920]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        as_strided_scatter: "bf16[81920]" = torch.ops.aten.as_strided_scatter.default(full, squeeze_1, _shape_param_2, _shape_param_3, 0);  full = squeeze_1 = _shape_param_2 = _shape_param_3 = None
        as_strided: "bf16[128, 640, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, _shape_param_4, _shape_param_5, 0);  as_strided_scatter = _shape_param_4 = _shape_param_5 = None
        expand: "bf16[128, 640, 7, 7]" = torch.ops.aten.expand.default(as_strided, _shape_param_6);  as_strided = _shape_param_6 = None
        div_1: "bf16[128, 640, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        sum_5: "bf16[640]" = torch.ops.aten.sum.dim_IntList(div_1, [0, 2, 3])
        convert_element_type_3: "f32[640]" = torch.ops.prims.convert_element_type.default(sum_5, torch.float32);  sum_5 = None
        return (sum_3, sum_4, div_1, convert_element_type_3)



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
