"""
Standalone repro captured via capture_hook.
Label: torchbench_llava_infer_000
Pattern hash: 7be7ad674239
Shape hash: 373ebedc
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 4096], f16), T([1, 512, 128], f16), T([1, 512, 128], f16), T([512, 4096], f16), S([1, 512, 4096]), S([1, 512, -1, 128]), S([1, 512, 4096]), S([1, 512, -1, 128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_217: "f16[512, 4096]", convert_element_type_4: "f16[1, 512, 128]", convert_element_type_5: "f16[1, 512, 128]", mm_218: "f16[512, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f16[1, 512, 4096]" = torch.ops.aten.view.default(mm_217, _shape_param_0);  mm_217 = _shape_param_0 = None
        view_default_1: "f16[1, 512, 32, 128]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        unsqueeze_default: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 1);  convert_element_type_4 = None
        mul_tensor: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default)
        slice_tensor: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 64, 9223372036854775807)
        neg_default: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None
        slice_tensor_1: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 64);  permute_default = None
        cat_default: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None
        unsqueeze_default_1: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 1);  convert_element_type_5 = None
        mul_tensor_1: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = None
        add_tensor: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        view_default_2: "f16[1, 512, 4096]" = torch.ops.aten.view.default(mm_218, _shape_param_2);  mm_218 = _shape_param_2 = None
        view_default_3: "f16[1, 512, 32, 128]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        permute_default_1: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        mul_tensor_2: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default_1, unsqueeze_default);  unsqueeze_default = None
        slice_tensor_2: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 9223372036854775807)
        neg_default_1: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None
        slice_tensor_3: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64);  permute_default_1 = None
        cat_default_1: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None
        mul_tensor_3: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_1);  cat_default_1 = unsqueeze_default_1 = None
        add_tensor_1: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        return (add_tensor, add_tensor_1)



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
