"""
Standalone repro captured via capture_hook.
Label: tritonbench_geglu_B4_T2048_H11008
Pattern hash: b374e9a57a10
Shape hash: f0b1e6c8
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4, 2048, 11008]", arg1_1: "bf16[4, 2048, 11008]"):
        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:185 in geglu_fn, code: return F.gelu(gate, approximate='tanh') * up
        convert_element_type_default: "f32[4, 2048, 11008]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        mul_tensor: "f32[4, 2048, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[4, 2048, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_default, convert_element_type_default)
        mul_tensor_2: "f32[4, 2048, 11008]" = torch.ops.aten.mul.Tensor(mul_tensor_1, convert_element_type_default);  mul_tensor_1 = None
        mul_tensor_3: "f32[4, 2048, 11008]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.044715);  mul_tensor_2 = None
        add_tensor: "f32[4, 2048, 11008]" = torch.ops.aten.add.Tensor(convert_element_type_default, mul_tensor_3);  convert_element_type_default = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 2048, 11008]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[4, 2048, 11008]" = torch.ops.aten.tanh.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_1: "f32[4, 2048, 11008]" = torch.ops.aten.add.Tensor(tanh_default, 1);  tanh_default = None
        mul_tensor_5: "f32[4, 2048, 11008]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        convert_element_type_default_1: "bf16[4, 2048, 11008]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        mul_tensor_6: "bf16[4, 2048, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, arg1_1);  convert_element_type_default_1 = arg1_1 = None
        return mul_tensor_6


def _default_make_inputs():
    return [
    torch.randn([4, 2048, 11008], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 2048, 11008], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
