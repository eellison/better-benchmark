"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: f24f665135f6
Shape hash: 2f9fa5d4
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
    def forward(self, arg0_1: "bf16[2048, 768]", arg1_1: "bf16[16, 768]", arg2_1: "b8[16, 128, 768]", arg3_1: "b8[16, 128, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[16, 128, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_1: "f32[16, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        full: "f32[16, 128, 768]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        select_scatter: "f32[16, 128, 768]" = torch.ops.aten.select_scatter.default(full, convert_element_type_1, 1, 0);  full = convert_element_type_1 = None
        add: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type, select_scatter);  convert_element_type = select_scatter = None
        convert_element_type_2: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_1: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add, mul);  add = mul = None
        convert_element_type_3: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16)
        convert_element_type_4: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        mul_2: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_3: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, mul_2);  convert_element_type_3 = mul_2 = None
        view_1: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_3, _shape_param_2);  mul_3 = _shape_param_2 = None
        permute: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_1, [1, 0])
        sum_1: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32)
        view_2: "f32[768]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type_5: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_6: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        return (mul_1, view_1, permute, convert_element_type_6)



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
