"""
Standalone repro captured via capture_hook.
Label: torchbench_shufflenet_v2_x1_0_infer
Pattern hash: 3c17f0120a71
Shape hash: f47c0655
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
    def forward(self, arg0_1: "bf16[116]", arg1_1: "bf16[64, 116, 14, 14]", arg2_1: "bf16[116]", arg3_1: "bf16[116]", arg4_1: "bf16[116]", arg5_1: "bf16[116]", arg6_1: "bf16[64, 116, 14, 14]", arg7_1: "bf16[116]", arg8_1: "bf16[116]", arg9_1: "bf16[116]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "f32[116]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        unsqueeze: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        unsqueeze_1: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[64, 116, 14, 14]" = torch.ops.aten.sub.Tensor(arg1_1, unsqueeze_1);  arg1_1 = unsqueeze_1 = None
        convert_element_type_1: "f32[116]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
        sqrt: "f32[116]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[64, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "bf16[116, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_5: "bf16[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[64, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "bf16[116, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_7: "bf16[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[64, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_2: "bf16[64, 116, 14, 14]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        relu: "bf16[64, 116, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_2);  convert_element_type_2 = None
        convert_element_type_3: "f32[116]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        unsqueeze_8: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, -1);  convert_element_type_3 = None
        unsqueeze_9: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_1: "f32[64, 116, 14, 14]" = torch.ops.aten.sub.Tensor(arg6_1, unsqueeze_9);  arg6_1 = unsqueeze_9 = None
        convert_element_type_4: "f32[116]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        add_2: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_4, 1e-05);  convert_element_type_4 = None
        sqrt_1: "f32[116]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
        reciprocal_1: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_3: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_11: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_4: "f32[64, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12: "bf16[116, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_13: "bf16[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_5: "f32[64, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
        unsqueeze_14: "bf16[116, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_15: "bf16[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_3: "f32[64, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
        convert_element_type_5: "bf16[64, 116, 14, 14]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        relu_1: "bf16[64, 116, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_5);  convert_element_type_5 = None
        cat: "bf16[64, 232, 14, 14]" = torch.ops.aten.cat.default([relu, relu_1], 1);  relu = relu_1 = None
        view: "bf16[64, 2, 116, 14, 14]" = torch.ops.aten.view.default(cat, _shape_param_0);  cat = _shape_param_0 = None
        permute: "bf16[64, 116, 2, 14, 14]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3, 4]);  view = None
        clone: "bf16[64, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[64, 232, 14, 14]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        split = torch.ops.aten.split.Tensor(view_1, 116, 1);  view_1 = None
        getitem: "bf16[64, 116, 14, 14]" = split[0]
        getitem_1: "bf16[64, 116, 14, 14]" = split[1];  split = None
        return (getitem, getitem_1)



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
