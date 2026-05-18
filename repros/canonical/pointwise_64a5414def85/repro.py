"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g35
Pattern hash: 64a5414def85
Shape hash: 8e2f084a
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
    def forward(self, bmm_43: "f16[32, 1024, 64]", mm_183: "f16[4096, 2048]", arg51_1: "b8[4, 1024, 2048]", arg428_1: "b8[4, 1024, 2048]", full_2: "f16[]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_43, [4, 8, 1024, 64]);  bmm_43 = None
        permute_default: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, [4, 1024, 512]);  clone_default = None
        reshape_default_2: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_1, [4096, 512]);  reshape_default_1 = None
        reshape_default_3: "f16[4, 1024, 2048]" = torch.ops.aten.reshape.default(mm_183, [4, 1024, 2048]);  mm_183 = None
        convert_element_type_default: "f16[4, 1024, 2048]" = torch.ops.prims.convert_element_type.default(reshape_default_3, torch.float16);  reshape_default_3 = None
        convert_element_type_default_1: "f16[4, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg51_1, torch.float16);  arg51_1 = None
        mul_tensor: "f16[4, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_1: "f16[4, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  convert_element_type_default = mul_tensor = None
        where_self: "f16[4, 1024, 2048]" = torch.ops.aten.where.self(arg428_1, full_2, mul_tensor_1);  arg428_1 = full_2 = mul_tensor_1 = None
        reshape_default_4: "f16[4096, 2048]" = torch.ops.aten.reshape.default(where_self, [4096, 2048]);  where_self = None
        return (reshape_default_2, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [4, 1024, 2048], dtype=torch.bool, device='cuda'),
    torch.randint(0, 2, [4, 1024, 2048], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
