"""
Standalone repro captured via capture_hook.
Label: vllm_meta-llama_Llama-3.2-1B_000
Pattern hash: 352e97c8de09
Shape hash: 2b7363a6
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
_shapes_config = "(T([2048, 2048], bf16), T([1, 512, 64], bf16), T([1, 512, 64], bf16), T([2048, 512], bf16), S([4, 512, 2048]), S([4, 512, -1, 64]), S([4, 512, 512]), S([4, 512, -1, 64]), S([4, 8, 4, 512, 64]), S([4, 32, 512, 64]))"

class Repro(torch.nn.Module):
    def forward(self, mm_105: "bf16[2048, 2048]", convert_element_type_3: "bf16[1, 512, 64]", convert_element_type_4: "bf16[1, 512, 64]", mm_106: "bf16[2048, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(mm_105, _shape_param_0);  mm_105 = _shape_param_0 = None
        view_default_1: "bf16[4, 512, 32, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "bf16[4, 32, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        unsqueeze_default: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
        mul_tensor: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default)
        slice_tensor: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 32, 9223372036854775807)
        neg_default: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None
        slice_tensor_1: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 32);  permute_default = None
        cat_default: "bf16[4, 32, 512, 64]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None
        unsqueeze_default_1: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 1);  convert_element_type_4 = None
        mul_tensor_1: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = None
        add_tensor: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        view_default_2: "bf16[4, 512, 512]" = torch.ops.aten.view.default(mm_106, _shape_param_2);  mm_106 = _shape_param_2 = None
        view_default_3: "bf16[4, 512, 8, 64]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        permute_default_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        mul_tensor_2: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(permute_default_1, unsqueeze_default);  unsqueeze_default = None
        slice_tensor_2: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 32, 9223372036854775807)
        neg_default_1: "bf16[4, 8, 512, 32]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None
        slice_tensor_3: "bf16[4, 8, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 32);  permute_default_1 = None
        cat_default_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None
        mul_tensor_3: "bf16[4, 8, 512, 64]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_1);  cat_default_1 = unsqueeze_default_1 = None
        add_tensor_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_2: "bf16[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 2);  add_tensor_1 = None
        expand_default: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_default_2, _shape_param_4);  unsqueeze_default_2 = _shape_param_4 = None
        clone_default: "bf16[4, 8, 4, 512, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_4: "bf16[4, 32, 512, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        return (add_tensor, view_default_4)



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
