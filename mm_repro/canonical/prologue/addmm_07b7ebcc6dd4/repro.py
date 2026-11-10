"""Curated MM repro. Pattern hash: 07b7ebcc6dd4."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[50272, 768]", arg1_1: "i64[4, 2048]", arg2_1: "f32[4, 2048]", arg3_1: "bf16[2050, 768]", arg4_1: "bf16[768]", arg5_1: "bf16[768]", arg6_1: "bf16[768]", arg7_1: "bf16[768, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        embedding: "bf16[4, 2048, 768]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 1);  arg0_1 = arg1_1 = None
        sub: "f32[4, 2048]" = torch.ops.aten.sub.Tensor(arg2_1, 1);  arg2_1 = None
        convert_element_type: "i64[4, 2048]" = torch.ops.prims.convert_element_type.default(sub, torch.int64);  sub = None
        add: "i64[4, 2048]" = torch.ops.aten.add.Tensor(convert_element_type, 2);  convert_element_type = None
        embedding_1: "bf16[4, 2048, 768]" = torch.ops.aten.embedding.default(arg3_1, add);  arg3_1 = add = None
        add_1: "bf16[4, 2048, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        convert_element_type_1: "f32[4, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
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
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_0);  convert_element_type_2 = _shape_param_0 = None
        addmm: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg6_1, view, arg7_1);  arg6_1 = view = arg7_1 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
