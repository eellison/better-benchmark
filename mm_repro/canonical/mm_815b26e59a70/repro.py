"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/repvgg_a2/full_graph_001.py
Pattern hash: 815b26e59a70
Shape hash: 0db7227a
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 1408]", arg2_1: "bf16[128, 1408, 7, 7]", arg3_1: "f32[1, 1408, 1, 1]", arg4_1: "f32[1, 1408, 1, 1]", arg5_1: "f32[1408]", arg6_1: "f32[1408]", arg7_1: "bf16[128, 1408, 7, 7]", arg8_1: "f32[1, 1408, 1, 1]", arg9_1: "f32[1, 1408, 1, 1]", arg10_1: "f32[1408]", arg11_1: "f32[1408]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        mm: "bf16[128, 1408]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 1408, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        squeeze: "bf16[128, 1408, 1]" = torch.ops.aten.squeeze.dim(view, 3);  view = None
        squeeze_1: "bf16[128, 1408]" = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
        full: "bf16[180224]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        as_strided_scatter: "bf16[180224]" = torch.ops.aten.as_strided_scatter.default(full, squeeze_1, _shape_param_2, _shape_param_3, 0);  full = squeeze_1 = _shape_param_2 = _shape_param_3 = None
        as_strided: "bf16[128, 1408, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, _shape_param_4, _shape_param_5, 0);  as_strided_scatter = _shape_param_4 = _shape_param_5 = None
        expand: "bf16[128, 1408, 7, 7]" = torch.ops.aten.expand.default(as_strided, _shape_param_6);  as_strided = _shape_param_6 = None
        div: "bf16[128, 1408, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        sub: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(arg2_1, arg3_1)
        mul: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = arg4_1 = None
        unsqueeze: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_1: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_3: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        sub_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(arg7_1, arg8_1)
        mul_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_1, arg9_1);  sub_1 = arg9_1 = None
        unsqueeze_4: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_5: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_2, unsqueeze_5);  mul_2 = unsqueeze_5 = None
        unsqueeze_6: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_7: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_3, unsqueeze_7);  mul_3 = unsqueeze_7 = None
        convert_element_type_1: "bf16[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        add_2: "bf16[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        relu: "bf16[128, 1408, 7, 7]" = torch.ops.aten.relu.default(add_2);  add_2 = None
        le: "b8[128, 1408, 7, 7]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 1408, 7, 7]" = torch.ops.aten.where.self(le, full_1, div);  le = full_1 = div = None
        convert_element_type_2: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze_2: "f32[1408]" = torch.ops.aten.squeeze.dims(arg8_1, [0, 2, 3]);  arg8_1 = None
        unsqueeze_8: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(squeeze_2, 0);  squeeze_2 = None
        unsqueeze_9: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None
        unsqueeze_10: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 3);  unsqueeze_9 = None
        convert_element_type_3: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        sub_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_3, unsqueeze_10);  convert_element_type_3 = unsqueeze_10 = None
        mul_4: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_2);  sub_2 = None
        squeeze_3: "f32[1408]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        unsqueeze_11: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_12: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 2);  unsqueeze_11 = None
        unsqueeze_13: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 3);  unsqueeze_12 = None
        convert_element_type_4: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_4, unsqueeze_13);  convert_element_type_4 = unsqueeze_13 = None
        mul_5: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_3);  sub_3 = None
        return (convert_element_type_2, mul_4, mul_5)



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
