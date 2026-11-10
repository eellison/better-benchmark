"""Curated MM repro. Pattern hash: 5bbc05be2ec1."""
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
    def forward(self, arg0_1: "bf16[201088, 2880]", arg1_1: "i64[1, 1000]", arg2_1: "bf16[2880]", arg3_1: "bf16[512]", arg4_1: "bf16[2880, 512]", _shape_param_0):
        # No stacktrace found for following nodes
        embedding: "bf16[1, 1000, 2880]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 199999);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32);  embedding = None
        pow_1: "f32[1, 1000, 2880]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        mul_1: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(arg2_1, mul);  arg2_1 = mul = None
        convert_element_type_1: "bf16[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        view: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        addmm: "bf16[1000, 512]" = torch.ops.aten.addmm.default(arg3_1, view, arg4_1);  arg3_1 = view = arg4_1 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
