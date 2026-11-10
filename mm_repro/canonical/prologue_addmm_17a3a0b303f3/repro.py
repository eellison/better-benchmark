"""Curated MM repro. Pattern hash: 17a3a0b303f3."""
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
    def forward(self, arg0_1: "bf16[8008, 2560]", arg1_1: "i64[32, 128]", arg2_1: "bf16[128, 2560]", arg3_1: "bf16[2560]", arg4_1: "bf16[2560]", arg5_1: "bf16[2560]", arg6_1: "bf16[2560, 2560]", _shape_param_0):
        # No stacktrace found for following nodes
        embedding: "bf16[32, 128, 2560]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        mul: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        embedding_1: "bf16[128, 2560]" = torch.ops.aten.embedding.default(arg2_1, add);  arg2_1 = add = None
        add_1: "bf16[32, 128, 2560]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
        convert_element_type: "f32[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean[0]
        getitem_1: "f32[32, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_2: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_1: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_3: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        convert_element_type_1: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[4096, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        addmm: "bf16[4096, 2560]" = torch.ops.aten.addmm.default(arg5_1, view, arg6_1);  arg5_1 = view = arg6_1 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
