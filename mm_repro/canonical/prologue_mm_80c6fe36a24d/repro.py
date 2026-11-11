"""Curated MM repro. Pattern hash: 80c6fe36a24d."""
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
    def forward(self, arg0_1: "f32[50400, 4096]", arg1_1: "i64[1, 128]", arg2_1: "f32[4096]", arg3_1: "f32[4096]", arg4_1: "bf16[4096, 4096]", _shape_param_0):
        # No stacktrace found for following nodes
        embedding: "f32[1, 128, 4096]" = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        var_mean = torch.ops.aten.var_mean.correction(embedding, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean[0]
        getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(embedding, getitem_1);  embedding = getitem_1 = None
        mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        convert_element_type: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(view, arg4_1);  view = arg4_1 = None
        return mm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
