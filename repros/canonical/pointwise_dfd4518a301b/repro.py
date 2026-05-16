"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g174
Pattern hash: dfd4518a301b
Shape hash: 9bb19270
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_134: "f16[1904, 3072]", arg37_1: "f16[1904, 3072]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 476, 3072]" = torch.ops.aten.reshape.default(mm_134, [4, 476, 3072]);  mm_134 = None
        reshape_default_1: "f16[4, 476, 3072]" = torch.ops.aten.reshape.default(arg37_1, [4, 476, 3072]);  arg37_1 = None
        mul_tensor: "f16[4, 476, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        mul_tensor_1: "f16[4, 476, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  mul_tensor = None
        div_tensor: "f16[4, 476, 3072]" = torch.ops.aten.div.Tensor(reshape_default_1, 1.4142135623730951);  reshape_default_1 = None
        erf_default: "f16[4, 476, 3072]" = torch.ops.aten.erf.default(div_tensor)
        add_tensor: "f16[4, 476, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1.0);  erf_default = None
        mul_tensor_2: "f16[4, 476, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, add_tensor);  reshape_default = add_tensor = None
        convert_element_type_default: "f32[4, 476, 3072]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float32);  div_tensor = None
        pow_tensor_scalar: "f32[4, 476, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2);  convert_element_type_default = None
        neg_default: "f32[4, 476, 3072]" = torch.ops.aten.neg.default(pow_tensor_scalar);  pow_tensor_scalar = None
        exp_default: "f32[4, 476, 3072]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        mul_scalar: "f32[4, 476, 3072]" = torch.ops.aten.mul.Scalar(exp_default, 1.1283791670955126);  exp_default = None
        mul_tensor_3: "f32[4, 476, 3072]" = torch.ops.aten.mul.Tensor(mul_scalar, mul_tensor_1);  mul_scalar = mul_tensor_1 = None
        convert_element_type_default_1: "f16[4, 476, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.float16);  mul_tensor_3 = None
        div_tensor_1: "f16[4, 476, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_default_1, 1.4142135623730951);  convert_element_type_default_1 = None
        mul_tensor_4: "f16[4, 476, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.5);  mul_tensor_2 = None
        add_tensor_1: "f16[4, 476, 3072]" = torch.ops.aten.add.Tensor(div_tensor_1, mul_tensor_4);  div_tensor_1 = mul_tensor_4 = None
        reshape_default_2: "f16[1904, 3072]" = torch.ops.aten.reshape.default(add_tensor_1, [1904, 3072]);  add_tensor_1 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
