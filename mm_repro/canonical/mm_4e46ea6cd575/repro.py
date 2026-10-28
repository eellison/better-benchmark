"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/mobilenetv3_large_100/full_graph_001.py
Pattern hash: 4e46ea6cd575
Shape hash: 33f6c945
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
    def forward(self, arg0_1: "bf16[512, 1000]", arg1_1: "bf16[1000, 1280]", arg2_1: "bf16[512, 1280, 1, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[512, 1280]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[512, 1280, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[512, 1280, 1, 1]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_1: "f32[512, 1280, 1, 1]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        le: "b8[512, 1280, 1, 1]" = torch.ops.aten.le.Scalar(convert_element_type_1, -3)
        lt: "b8[512, 1280, 1, 1]" = torch.ops.aten.lt.Scalar(convert_element_type_1, 3)
        div: "f32[512, 1280, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_1, 3);  convert_element_type_1 = None
        add: "f32[512, 1280, 1, 1]" = torch.ops.aten.add.Tensor(div, 0.5);  div = None
        mul: "f32[512, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type, add);  add = None
        where: "f32[512, 1280, 1, 1]" = torch.ops.aten.where.self(lt, mul, convert_element_type);  lt = mul = convert_element_type = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[512, 1280, 1, 1]" = torch.ops.aten.where.self(le, full, where);  le = full = where = None
        convert_element_type_2: "bf16[512, 1280, 1, 1]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        return convert_element_type_2



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
