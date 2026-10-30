"""Curated MM repro. Pattern hash: 7fe5be0f6b2c."""
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
    def forward(self, arg0_1: "bf16[30000, 128]", arg1_1: "i64[8, 512]", arg2_1: "i64[1, 512]", arg3_1: "i64[1, 512]", arg4_1: "bf16[2, 128]", arg5_1: "bf16[512, 128]", arg6_1: "bf16[128]", arg7_1: "bf16[128]", arg8_1: "bf16[128, 4096]", arg9_1: "bf16[4096]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        embedding: "bf16[8, 512, 128]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        gather: "i64[1, 512]" = torch.ops.aten.gather.default(arg2_1, 1, arg3_1);  arg2_1 = None
        expand_1: "i64[8, 512]" = torch.ops.aten.expand.default(gather, _shape_param_0);  gather = _shape_param_0 = None
        embedding_1: "bf16[8, 512, 128]" = torch.ops.aten.embedding.default(arg4_1, expand_1);  arg4_1 = expand_1 = None
        add: "bf16[8, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2: "bf16[1, 512, 128]" = torch.ops.aten.embedding.default(arg5_1, arg3_1);  arg5_1 = arg3_1 = None
        add_1: "bf16[8, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        convert_element_type: "f32[8, 512, 128]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean[0]
        getitem_1: "f32[8, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_2: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        convert_element_type_1: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[4096, 128]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        addmm: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg9_1, view, arg8_1);  arg9_1 = view = arg8_1 = None
        view_1: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_2);  _shape_param_2 = None
        return (view_1, addmm, addmm, addmm)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
