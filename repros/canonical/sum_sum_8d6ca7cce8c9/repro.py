"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: 8d6ca7cce8c9
Shape hash: 0c475f9a
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
    def forward(self, arg0_1: "bf16[32, 6, 512, 64]", arg1_1: "bf16[16384, 384]", arg2_1: "bf16[32, 384, 512]", arg3_1: "f32[384, 1]", arg4_1: "bf16[32, 512, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        permute: "bf16[32, 512, 6, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[32, 512, 384]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        add: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        convert_element_type: "bf16[32, 384, 512]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        permute_1: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(convert_element_type, [0, 2, 1]);  convert_element_type = None
        mul: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(view, permute_1);  permute_1 = None
        mul_1: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(view, arg4_1);  view = arg4_1 = None
        clone: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[32, 512, 384]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        add_1: "bf16[32, 512, 384]" = torch.ops.aten.add.Tensor(mul, view_1);  mul = view_1 = None
        view_2: "bf16[16384, 384]" = torch.ops.aten.view.default(add_1, _shape_param_2);  add_1 = _shape_param_2 = None
        permute_2: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_1: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[384]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type_1: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_2: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        permute_3: "bf16[32, 384, 512]" = torch.ops.aten.permute.default(mul_1, [0, 2, 1]);  mul_1 = None
        sum_2: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_3, [0, 2], True, dtype = torch.float32)
        view_4: "f32[384, 1]" = torch.ops.aten.view.default(sum_2, _shape_param_4);  sum_2 = _shape_param_4 = None
        return (view_2, permute_2, convert_element_type_2, permute_3, view_4)



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
