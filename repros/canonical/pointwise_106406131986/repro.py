"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_infer_000
Pattern hash: 106406131986
Shape hash: 9115f773
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 384, 3000], f16))"

class Repro(torch.nn.Module):
    def forward(self, convolution: "f16[8, 384, 3000]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[8, 384, 3000]" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        mul_tensor: "f32[8, 384, 3000]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[8, 384, 3000]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[8, 384, 3000]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[8, 384, 3000]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[8, 384, 3000]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1: "f16[8, 384, 3000]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float16);  mul_tensor_2 = None
        return convert_element_type_default_1

def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)

def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
