"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: b3749c8f333c
Shape hash: 2dcdf79e
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
_shapes_config = "(T([1, 6, 6144, 128], bf16, stride=(4718592, 128, 768, 1)), T([1, 6144, 6, 128], bf16), T([1, 6144, 6], bf16), S([1, 6144, 6]), S([6144, 6]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_19: "bf16[1, 6, 6144, 128]", view_354: "bf16[1, 6144, 6, 128]", sigmoid: "bf16[1, 6144, 6]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_default: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_19, [0, 2, 1, 3]);  getitem_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        mul_tensor: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(view_354, permute_default);  view_354 = permute_default = None
        sum_dim_int_list: "bf16[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True);  mul_tensor = None
        reshape_default: "bf16[1, 6144, 6]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        convert_element_type_default: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        convert_element_type_default_1: "f32[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(sigmoid, torch.float32);  sigmoid = None
        sub_tensor: "f32[1, 6144, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_default_1)
        mul_tensor_1: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, sub_tensor);  convert_element_type_default_1 = sub_tensor = None
        mul_tensor_2: "f32[1, 6144, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_1);  convert_element_type_default = mul_tensor_1 = None
        convert_element_type_default_2: "bf16[1, 6144, 6]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        reshape_default_1: "bf16[6144, 6]" = torch.ops.aten.reshape.default(convert_element_type_default_2, _shape_param_1);  convert_element_type_default_2 = _shape_param_1 = None
        permute_default_1: "bf16[6, 6144]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0]);  reshape_default_1 = None
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
