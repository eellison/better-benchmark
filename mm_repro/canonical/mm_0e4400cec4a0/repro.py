"""
Standalone repro captured via capture_hook.
Label: repros/models/torchbench/train/shufflenet_v2_x1_0/full_graph_001.py
Pattern hash: 0e4400cec4a0
Shape hash: 55a59aa7
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 1024]", arg2_1: "bf16[128, 1024, 7, 7]", arg3_1: "f32[1, 1024, 1, 1]", arg4_1: "f32[1, 1024, 1, 1]", arg5_1: "f32[1024]", arg6_1: "f32[1024]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[128, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        unsqueeze: "bf16[128, 1024, 1]" = torch.ops.aten.unsqueeze.default(mm, 2);  mm = None
        unsqueeze_1: "bf16[128, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 3);  unsqueeze = None
        expand: "bf16[128, 1024, 7, 7]" = torch.ops.aten.expand.default(unsqueeze_1, _shape_param_0);  unsqueeze_1 = _shape_param_0 = None
        div: "bf16[128, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        sub: "f32[128, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(arg2_1, arg3_1)
        mul: "f32[128, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = arg4_1 = None
        unsqueeze_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[128, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_3);  mul = unsqueeze_3 = None
        unsqueeze_4: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_5: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        add: "f32[128, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        convert_element_type: "bf16[128, 1024, 7, 7]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        relu: "bf16[128, 1024, 7, 7]" = torch.ops.aten.relu.default(convert_element_type);  convert_element_type = None
        le: "b8[128, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 1024, 7, 7]" = torch.ops.aten.where.self(le, full, div);  le = full = div = None
        convert_element_type_1: "f32[128, 1024, 7, 7]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze: "f32[1024]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        unsqueeze_6: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_7: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        convert_element_type_2: "f32[128, 1024, 7, 7]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub_1: "f32[128, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_8);  convert_element_type_2 = unsqueeze_8 = None
        mul_2: "f32[128, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub_1);  sub_1 = None
        return (convert_element_type_1, mul_2)



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
