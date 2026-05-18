"""
Standalone repro captured via capture_hook.
Label: genai_patterns
Pattern hash: d1ac77b5eb39
Shape hash: 17eb7221
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_1: "bf16[4, 2048, 4096]", getitem: "bf16[4, 2048, 4096]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:102 in geglu_pattern, code: return x1 * F.gelu(x2)
        convert_element_type_default: "f32[4, 2048, 4096]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
        mul_tensor: "f32[4, 2048, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[4, 2048, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[4, 2048, 4096]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[4, 2048, 4096]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[4, 2048, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1: "bf16[4, 2048, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        mul_tensor_3: "bf16[4, 2048, 4096]" = torch.ops.aten.mul.Tensor(getitem, convert_element_type_default_1);  getitem = convert_element_type_default_1 = None
        return mul_tensor_3


def _default_make_inputs():
    return [
    torch.randn(67104768, dtype=torch.bfloat16, device='cuda').as_strided([4, 2048, 4096], [16777216, 8192, 1]),  # getitem_1
    torch.randn(67104768, dtype=torch.bfloat16, device='cuda').as_strided([4, 2048, 4096], [16777216, 8192, 1]),  # getitem
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
