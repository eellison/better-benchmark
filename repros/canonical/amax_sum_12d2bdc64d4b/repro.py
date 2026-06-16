"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train
Pattern hash: 12d2bdc64d4b
Shape hash: 385d7a56
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
    def forward(self, arg0_1: "bf16[512, 128, 128]", arg1_1: "b8[1, 1, 2048, 2048]", arg2_1: "bf16[]", arg3_1: "f32[32, 1, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        slice_1: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg1_1, 2, 0, 128);  arg1_1 = None
        slice_2: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 128);  slice_1 = None
        where: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, view, arg2_1);  slice_2 = view = arg2_1 = None
        add: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where, arg3_1);  where = arg3_1 = None
        amax: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add, [-1], True)
        sub: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add, amax);  add = amax = None
        exp: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16)
        expand: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        view_1: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand, _shape_param_2);  expand = _shape_param_2 = None
        permute: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1])
        return (div, view_1, permute)



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
