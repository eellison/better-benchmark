"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_infer
Pattern hash: fbc9e58f2e01
Shape hash: 801db743
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
    def forward(self, arg0_1: "bf16[50272, 768]", arg1_1: "i64[4, 2048]", arg2_1: "f32[4, 2048]", arg3_1: "bf16[2050, 768]", arg4_1: "bf16[768]", arg5_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        embedding: "bf16[4, 2048, 768]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 1);  arg0_1 = arg1_1 = None
        sub: "f32[4, 2048]" = torch.ops.aten.sub.Tensor(arg2_1, 1);  arg2_1 = None
        convert_element_type: "i64[4, 2048]" = torch.ops.prims.convert_element_type.default(sub, torch.int64);  sub = None
        add: "i64[4, 2048]" = torch.ops.aten.add.Tensor(convert_element_type, 2);  convert_element_type = None
        embedding_1: "bf16[4, 2048, 768]" = torch.ops.aten.embedding.default(arg3_1, add);  arg3_1 = add = None
        add_1: "bf16[4, 2048, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        convert_element_type_1: "f32[4, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_1, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 2048, 1]" = var_mean[0]
        getitem_1: "f32[4, 2048, 1]" = var_mean[1];  var_mean = None
        sub_1: "f32[4, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_1, getitem_1);  convert_element_type_1 = getitem_1 = None
        add_2: "f32[4, 2048, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 2048, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_1: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_3: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_2: "bf16[4, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_0);  _shape_param_0 = None
        view_1: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_1);  _shape_param_1 = None
        view_2: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        return (add_1, view, view_1, view_2)



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
