"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_train
Pattern hash: 29c472896711
Shape hash: c7ec772c
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
    def forward(self, arg0_1: "b8[32, 1280]", arg1_1: "bf16[32, 1280]", arg2_1: "bf16[32, 1280]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        div: "bf16[32, 1280]" = torch.ops.aten.div.Scalar(convert_element_type, 0.8);  convert_element_type = None
        mul: "bf16[32, 1280]" = torch.ops.aten.mul.Tensor(arg1_1, div);  arg1_1 = div = None
        convert_element_type_1: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_2: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        le: "b8[32, 1280]" = torch.ops.aten.le.Scalar(convert_element_type_2, -3)
        lt: "b8[32, 1280]" = torch.ops.aten.lt.Scalar(convert_element_type_2, 3)
        div_1: "f32[32, 1280]" = torch.ops.aten.div.Tensor(convert_element_type_2, 3);  convert_element_type_2 = None
        add: "f32[32, 1280]" = torch.ops.aten.add.Tensor(div_1, 0.5);  div_1 = None
        mul_1: "f32[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_1, add);  add = None
        where: "f32[32, 1280]" = torch.ops.aten.where.self(lt, mul_1, convert_element_type_1);  lt = mul_1 = convert_element_type_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 1280]" = torch.ops.aten.where.self(le, full, where);  le = where = None
        convert_element_type_3: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        permute: "bf16[1280, 32]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0])
        sum_1: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0], True, dtype = torch.float32)
        view: "f32[1280]" = torch.ops.aten.view.default(sum_1, _shape_param_0);  sum_1 = _shape_param_0 = None
        convert_element_type_4: "bf16[1280]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_5: "f32[1280]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        return (full, convert_element_type_3, permute, convert_element_type_5)



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
