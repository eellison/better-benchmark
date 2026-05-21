"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: 0b5de82d84ef
Shape hash: ad78d118
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
_shapes_config = "(T([6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 1], f32), T([1, 6144, 768], bf16), S([1, 6144, 768]), S([6144, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_165: "bf16[6144, 768]", add_10: "bf16[1, 6144, 768]", rsqrt_4: "f32[1, 6144, 1]", add_273: "bf16[1, 6144, 768]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        reshape_default: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_165, _shape_param_0);  mm_165 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        convert_element_type_default_1: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32);  add_10 = None
        mul_tensor: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_4);  convert_element_type_default_1 = None
        mul_tensor_1: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, convert_element_type_default)
        sum_dim_int_list: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        div_tensor: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_tensor, 768);  mul_tensor = None
        mul_tensor_2: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sum_dim_int_list);  div_tensor = sum_dim_int_list = None
        sub_tensor: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, mul_tensor_2);  convert_element_type_default = mul_tensor_2 = None
        mul_tensor_3: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_4);  sub_tensor = rsqrt_4 = None
        convert_element_type_default_2: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        add_tensor: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_273, convert_element_type_default_2);  add_273 = convert_element_type_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        reshape_default_1: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        return reshape_default_1



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
