"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: 894e623ad263
Shape hash: 943c9ed8
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
    def forward(self, arg0_1: "bf16[128, 2304]", arg1_1: "bf16[128, 2304, 7, 7]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view: "bf16[128, 2304, 1, 1]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        squeeze: "bf16[128, 2304, 1]" = torch.ops.aten.squeeze.dim(view, 3);  view = None
        squeeze_1: "bf16[128, 2304]" = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
        full: "bf16[294912]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        as_strided_scatter: "bf16[294912]" = torch.ops.aten.as_strided_scatter.default(full, squeeze_1, _shape_param_2, _shape_param_3, 0);  full = squeeze_1 = _shape_param_2 = _shape_param_3 = None
        as_strided: "bf16[128, 2304, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, _shape_param_4, _shape_param_5, 0);  as_strided_scatter = _shape_param_4 = _shape_param_5 = None
        expand: "bf16[128, 2304, 7, 7]" = torch.ops.aten.expand.default(as_strided, _shape_param_6);  as_strided = _shape_param_6 = None
        div: "bf16[128, 2304, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        convert_element_type: "f32[128, 2304, 7, 7]" = torch.ops.prims.convert_element_type.default(div, torch.float32);  div = None
        convert_element_type_1: "f32[128, 2304, 7, 7]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        sigmoid: "f32[128, 2304, 7, 7]" = torch.ops.aten.sigmoid.default(convert_element_type_1)
        mul: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = None
        sub: "f32[128, 2304, 7, 7]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_1: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub);  convert_element_type_1 = sub = None
        add: "f32[128, 2304, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, 1);  mul_1 = None
        mul_2: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_2: "bf16[128, 2304, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        sum_1: "bf16[2304]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 2, 3])
        convert_element_type_3: "f32[2304]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        return (convert_element_type_2, convert_element_type_3)



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
