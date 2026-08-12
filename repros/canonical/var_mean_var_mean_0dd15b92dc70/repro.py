"""
Standalone repro captured via capture_hook.
Label: hf_MBartForCausalLM_infer
Pattern hash: 0dd15b92dc70
Shape hash: 7a83b18a
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
    def forward(self, arg0_1: "bf16[50265, 1024]", arg1_1: "i64[8, 1024]", arg2_1: "bf16[1026, 1024]", arg3_1: "bf16[1024]", arg4_1: "bf16[1024]", arg5_1: "bf16[1024]", arg6_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        embedding: "bf16[8, 1024, 1024]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 1);  arg0_1 = arg1_1 = None
        mul: "bf16[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        add_1: "i64[1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze, 2);  unsqueeze = None
        embedding_1: "bf16[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg2_1, add_1);  arg2_1 = add_1 = None
        add_2: "bf16[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
        convert_element_type: "f32[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_2, torch.float32);  add_2 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        sub: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_3: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_4: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        convert_element_type_1: "bf16[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        convert_element_type_2: "f32[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_2, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 1024, 1]" = var_mean_1[0]
        getitem_3: "f32[8, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_3);  convert_element_type_2 = getitem_3 = None
        add_5: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_3: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_4: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
        add_6: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_4, arg6_1);  mul_4 = arg6_1 = None
        convert_element_type_3: "bf16[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        view: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_0);  _shape_param_0 = None
        view_1: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_1);  _shape_param_1 = None
        view_2: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_2);  convert_element_type_3 = _shape_param_2 = None
        return (convert_element_type_1, view, view_1, view_2)



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
