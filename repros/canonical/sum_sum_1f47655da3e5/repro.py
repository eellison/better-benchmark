"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 1f47655da3e5
Shape hash: 09c2ade4
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
    def forward(self, arg0_1: "bf16[128, 768]", arg1_1: "f32[128, 768, 7, 7]", arg2_1: "f32[768]", arg3_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "f32[128, 768, 1, 1]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        squeeze: "f32[128, 768, 1]" = torch.ops.aten.squeeze.dim(view, 3);  view = None
        squeeze_1: "f32[128, 768]" = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
        full: "f32[98304]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        as_strided_scatter: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full, squeeze_1, _shape_param_2, _shape_param_3, 0);  full = squeeze_1 = _shape_param_2 = _shape_param_3 = None
        as_strided: "f32[128, 768, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, _shape_param_4, _shape_param_5, 0);  as_strided_scatter = _shape_param_4 = _shape_param_5 = None
        expand: "f32[128, 768, 7, 7]" = torch.ops.aten.expand.default(as_strided, _shape_param_6);  as_strided = _shape_param_6 = None
        div: "f32[128, 768, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        sum_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(div, [0, 2, 3])
        mul: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(div, arg1_1)
        sum_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul, [0, 2, 3]);  mul = None
        mul_1: "f32[768]" = torch.ops.aten.mul.Tensor(sum_1, 0.00015943877551020407)
        unsqueeze: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1, 0);  mul_1 = None
        unsqueeze_1: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_2: "f32[768]" = torch.ops.aten.mul.Tensor(sum_2, 0.00015943877551020407)
        mul_3: "f32[768]" = torch.ops.aten.mul.Tensor(arg2_1, arg2_1)
        mul_4: "f32[768]" = torch.ops.aten.mul.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        unsqueeze_3: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_4, 0);  mul_4 = None
        unsqueeze_4: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_5: "f32[768]" = torch.ops.aten.mul.Tensor(arg2_1, arg3_1);  arg3_1 = None
        unsqueeze_6: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_7: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_6: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(arg1_1, unsqueeze_5);  arg1_1 = unsqueeze_5 = None
        sub: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(div, mul_6);  div = mul_6 = None
        sub_1: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub, unsqueeze_2);  sub = unsqueeze_2 = None
        mul_7: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_8);  sub_1 = unsqueeze_8 = None
        mul_8: "f32[768]" = torch.ops.aten.mul.Tensor(sum_2, arg2_1);  sum_2 = arg2_1 = None
        convert_element_type_1: "bf16[128, 768, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16)
        return (sum_1, mul_7, mul_8, convert_element_type_1)



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
