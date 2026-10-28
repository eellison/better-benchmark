"""
Standalone repro captured via capture_hook.
Label: repros/models/torchbench/train/mnasnet1_0/full_graph_001.py
Pattern hash: 08deccedc20e
Shape hash: 59f10daa
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
    def forward(self, arg0_1: "bf16[32, 1000]", arg1_1: "bf16[1000, 1280]", arg2_1: "b8[32, 1280]", arg3_1: "bf16[32, 1280, 7, 7]", arg4_1: "f32[1, 1280, 1, 1]", arg5_1: "f32[1, 1280, 1, 1]", arg6_1: "f32[1280]", arg7_1: "f32[1280]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[32, 1280]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        div: "bf16[32, 1280]" = torch.ops.aten.div.Scalar(convert_element_type, 0.8);  convert_element_type = None
        mul: "bf16[32, 1280]" = torch.ops.aten.mul.Tensor(mm, div);  mm = div = None
        unsqueeze: "bf16[32, 1280, 1]" = torch.ops.aten.unsqueeze.default(mul, 2);  mul = None
        unsqueeze_1: "bf16[32, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 3);  unsqueeze = None
        expand: "bf16[32, 1280, 7, 7]" = torch.ops.aten.expand.default(unsqueeze_1, _shape_param_0);  unsqueeze_1 = _shape_param_0 = None
        div_1: "bf16[32, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        sub: "f32[32, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(arg3_1, arg4_1)
        mul_1: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = arg5_1 = None
        unsqueeze_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_2: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        unsqueeze_4: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_5: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        add: "f32[32, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_5);  mul_2 = unsqueeze_5 = None
        convert_element_type_1: "bf16[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        relu: "bf16[32, 1280, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None
        le: "b8[32, 1280, 7, 7]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[32, 1280, 7, 7]" = torch.ops.aten.where.self(le, full, div_1);  le = full = div_1 = None
        convert_element_type_2: "f32[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze: "f32[1280]" = torch.ops.aten.squeeze.dims(arg4_1, [0, 2, 3]);  arg4_1 = None
        unsqueeze_6: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_7: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        convert_element_type_3: "f32[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub_1: "f32[32, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_3, unsqueeze_8);  convert_element_type_3 = unsqueeze_8 = None
        mul_3: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_1);  sub_1 = None
        return (convert_element_type_2, mul_3)



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
