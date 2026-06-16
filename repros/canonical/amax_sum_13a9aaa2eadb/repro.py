"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_infer
Pattern hash: 13a9aaa2eadb
Shape hash: 0b7018c4
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
    def forward(self, arg0_1: "b8[1, 1, 2048, 2048]", arg1_1: "f32[512, 128, 128]", arg2_1: "bf16[32, 1, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        slice_1: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg0_1, 2, 0, 128);  arg0_1 = None
        slice_2: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 128);  slice_1 = None
        view: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        full: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, view, full);  slice_2 = view = full = None
        add: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where, arg2_1);  where = arg2_1 = None
        amax: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add, [-1], True)
        sub: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add, amax);  add = amax = None
        exp: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        view_1: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand, _shape_param_2);  expand = _shape_param_2 = None
        return view_1



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
