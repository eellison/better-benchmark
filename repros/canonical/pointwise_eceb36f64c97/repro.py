"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s8_g21
Pattern hash: eceb36f64c97
Shape hash: fe996859
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_1: "f16[256, 1]", permute_2: "f16[1024, 2]", permute_1: "f16[1024, 1024]"):
        # No stacktrace found for following nodes
        tanh_default: "f16[256, 1]" = torch.ops.aten.tanh.default(getitem_1);  getitem_1 = None
        add_tensor: "f16[256, 1]" = torch.ops.aten.add.Tensor(tanh_default, 1);  tanh_default = None
        mul_tensor: "f16[256, 1]" = torch.ops.aten.mul.Tensor(add_tensor, 6.0);  add_tensor = None
        add_tensor_1: "f16[256, 1]" = torch.ops.aten.add.Tensor(mul_tensor, -10.0);  mul_tensor = None
        convert_element_type_default: "f32[256, 1]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        exp_default: "f32[256, 1]" = torch.ops.aten.exp.default(convert_element_type_default);  convert_element_type_default = None
        permute_default: "f16[2, 1024]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_default_1: "f16[1024, 1024]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (exp_default, permute_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn(511, dtype=torch.float16, device='cuda').as_strided([256, 1], [2, 1]),  # getitem_1
    torch.randn(2048, dtype=torch.float16, device='cuda').as_strided([1024, 2], [1, 1024]),  # permute_2
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([1024, 1024], [1, 1024]),  # permute_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
