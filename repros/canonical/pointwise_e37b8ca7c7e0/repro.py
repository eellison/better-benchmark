"""
Standalone repro captured via capture_hook.
Label: genai_swiglu_decode
Pattern hash: e37b8ca72db3
Shape hash: c7e06629
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_1: "bf16[32, 1, 4096]", getitem: "bf16[32, 1, 4096]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[32, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
        neg_default: "f32[32, 1, 4096]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[32, 1, 4096]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[32, 1, 4096]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[32, 1, 4096]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "bf16[32, 1, 4096]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        mul_tensor: "bf16[32, 1, 4096]" = torch.ops.aten.mul.Tensor(getitem, convert_element_type_default_1);  getitem = convert_element_type_default_1 = None
        return mul_tensor


def _default_make_inputs():
    return [
    torch.randn(258048, dtype=torch.bfloat16, device='cuda').as_strided([32, 1, 4096], [8192, 8192, 1]),  # getitem_1
    torch.randn(258048, dtype=torch.bfloat16, device='cuda').as_strided([32, 1, 4096], [8192, 8192, 1]),  # getitem
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
