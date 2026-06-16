"""
Standalone repro captured via capture_hook.
Label: hf_RobertaForCausalLM_infer
Pattern hash: d1f8c258a72a
Shape hash: 975c5fbc
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
    def forward(self, arg0_1: "bf16[50265, 768]", arg1_1: "i64[32, 512]", arg2_1: "i64[1, 512]", arg3_1: "i64[32, 512]", arg4_1: "i32[32, 512]", arg5_1: "bf16[2, 768]", arg6_1: "bf16[512, 768]", arg7_1: "bf16[768]", arg8_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        embedding: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        convert_element_type: "i32[32, 512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.int32);  arg3_1 = None
        add: "i32[32, 512]" = torch.ops.aten.add.Tensor(convert_element_type, 0);  convert_element_type = None
        mul: "i32[32, 512]" = torch.ops.aten.mul.Tensor(add, arg4_1);  add = arg4_1 = None
        convert_element_type_1: "i64[32, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.int64);  mul = None
        add_1: "i64[32, 512]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
        gather: "i64[32, 512]" = torch.ops.aten.gather.default(expand, 1, add_1);  expand = None
        expand_1: "i64[32, 512]" = torch.ops.aten.expand.default(gather, _shape_param_1);  gather = _shape_param_1 = None
        embedding_1: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, expand_1);  arg5_1 = expand_1 = None
        add_2: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg6_1, add_1, 0);  arg6_1 = add_1 = None
        add_3: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(add_2, embedding_2);  add_2 = embedding_2 = None
        convert_element_type_2: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_1);  convert_element_type_2 = getitem_1 = None
        add_4: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, arg8_1);  mul_2 = arg8_1 = None
        convert_element_type_3: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        view: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_2);  _shape_param_2 = None
        view_1: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_3);  _shape_param_3 = None
        view_2: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_4);  _shape_param_4 = None
        return (convert_element_type_3, view, view_1, view_2)



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
