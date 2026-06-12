"""
Standalone repro captured via capture_hook.
Label: hf_openai/whisper-tiny_infer
Pattern hash: 63cad2383953
Shape hash: a5e05174
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
    def forward(self, arg0_1: "bf16[51865, 384]", arg1_1: "i64[1, 1]", arg2_1: "bf16[448, 384]", arg3_1: "bf16[384]", arg4_1: "bf16[384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        embedding: "bf16[1, 1, 384]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 50257);  arg0_1 = arg1_1 = None
        full: "i64[1, 1]" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index: "bf16[1, 1, 384]" = torch.ops.aten.index.Tensor(arg2_1, [full]);  arg2_1 = full = None
        add: "bf16[1, 1, 384]" = torch.ops.aten.add.Tensor(embedding, index);  embedding = index = None
        convert_element_type: "f32[1, 1, 384]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 1, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 1, 384]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[1, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_2: "f32[1, 1, 384]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type_1: "bf16[1, 1, 384]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view: "bf16[1, 384]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  _shape_param_0 = None
        view_1: "bf16[1, 384]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  _shape_param_1 = None
        view_2: "bf16[1, 384]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        return (add, view, view_1, view_2)



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
