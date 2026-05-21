"""
Standalone repro captured via capture_hook.
Label: vllm_meta-llama_Llama-3.2-1B_001
Pattern hash: e5772e195244
Shape hash: b513aaf5
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
_shapes_config = "(T([4, 32, 512, 64], bf16), T([1, 1, 512, 64], bf16), T([4, 32, 512, 64], bf16), T([1, 1, 512, 64], bf16), S([4, 512, 2048]), S([2048, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_45: "bf16[4, 32, 512, 64]", unsqueeze_6: "bf16[1, 1, 512, 64]", full_3: "bf16[4, 32, 512, 64]", unsqueeze_7: "bf16[1, 1, 512, 64]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mul_tensor: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_45, unsqueeze_6);  unsqueeze_6 = None
        slice_tensor: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor, 3, 0, 32)
        slice_tensor_1: "bf16[4, 32, 512, 32]" = torch.ops.aten.slice.Tensor(mul_tensor, 3, 32, 64);  mul_tensor = None
        neg_default: "bf16[4, 32, 512, 32]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None
        slice_scatter_default: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_3, neg_default, 3, 32, 9223372036854775807);  neg_default = None
        slice_scatter_default_1: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice_scatter.default(full_3, slice_tensor_1, 3, 0, 32);  full_3 = slice_tensor_1 = None
        add_tensor: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1);  slice_scatter_default = slice_scatter_default_1 = None
        mul_tensor_1: "bf16[4, 32, 512, 64]" = torch.ops.aten.mul.Tensor(getitem_45, unsqueeze_7);  getitem_45 = unsqueeze_7 = None
        add_tensor_1: "bf16[4, 32, 512, 64]" = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_1);  add_tensor = mul_tensor_1 = None
        permute_default: "bf16[4, 512, 32, 64]" = torch.ops.aten.permute.default(add_tensor_1, [0, 2, 1, 3]);  add_tensor_1 = None
        clone_default: "bf16[4, 512, 32, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        view_default_1: "bf16[2048, 2048]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default_1: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        return permute_default_1



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
