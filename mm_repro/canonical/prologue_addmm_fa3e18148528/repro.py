"""Curated MM repro. Pattern hash: fa3e18148528."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[50265, 1024]", arg1_1: "i64[8, 1024]", arg2_1: "bf16[1026, 1024]", arg3_1: "bf16[1024]", arg4_1: "bf16[1024]", arg5_1: "bf16[1024]", arg6_1: "bf16[1024, 1024]", _shape_param_0):
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
        view: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        addmm: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg5_1, view, arg6_1);  arg5_1 = view = arg6_1 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
