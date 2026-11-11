"""Curated MM repro. Pattern hash: c44f89301ec7."""
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
    def forward(self, arg0_1: "bf16[256000, 2304]", arg1_1: "i64[1, 1000]", arg2_1: "bf16[]", arg3_1: "bf16[2304]", arg4_1: "bf16[2304, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        embedding: "bf16[1, 1000, 2304]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        mul: "bf16[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(embedding, arg2_1);  embedding = arg2_1 = None
        convert_element_type: "f32[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        pow_1: "f32[1, 1000, 2304]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul_1: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        convert_element_type_1: "f32[2304]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        add_1: "f32[2304]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1.0);  convert_element_type_1 = None
        mul_2: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(mul_1, add_1);  mul_1 = add_1 = None
        convert_element_type_2: "bf16[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        view: "bf16[1000, 2304]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_0);  convert_element_type_2 = _shape_param_0 = None
        mm: "bf16[1000, 1024]" = torch.ops.aten.mm.default(view, arg4_1);  view = arg4_1 = None
        return mm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
