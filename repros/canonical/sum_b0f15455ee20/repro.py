"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: b0f15455ee20
Shape hash: 85417ef5
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
_shapes_config = "(T([64], f32), T([6144, 12], bf16), T([1, 6144, 768], bf16), T([6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 1], f32), T([1, 6144, 768], bf16), T([12, 2], f32), S([1, 6144, 12]), S([1, 6144, 768]), S([6144, 768]))"

class Repro(torch.nn.Module):
    def forward(self, primals_7: "f32[64]", mm_109: "bf16[6144, 12]", full_default_19: "bf16[1, 6144, 768]", mm_111: "bf16[6144, 768]", add_71: "bf16[1, 6144, 768]", rsqrt_25: "f32[1, 6144, 1]", add_194: "bf16[1, 6144, 768]", view_6: "f32[12, 2]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:913 in forward, code: skip_weights = self.scalars[: (len(self.blocks) // 2)]
        slice_tensor: "f32[6]" = torch.ops.aten.slice.Tensor(primals_7, 0, 0, 6);  primals_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        reshape_default: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_109, _shape_param_0);  mm_109 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_scatter_default: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, reshape_default, 2, 0, 12);  full_default_19 = reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        reshape_default_1: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_111, _shape_param_1);  mm_111 = _shape_param_1 = None
        add_tensor: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter_default, reshape_default_1);  slice_scatter_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        convert_element_type_default_1: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        mul_tensor: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_25);  convert_element_type_default_1 = None
        mul_tensor_1: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, convert_element_type_default)
        sum_dim_int_list: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        div_tensor: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_tensor, 768);  mul_tensor = None
        mul_tensor_2: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sum_dim_int_list);  div_tensor = sum_dim_int_list = None
        sub_tensor: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, mul_tensor_2);  convert_element_type_default = mul_tensor_2 = None
        mul_tensor_3: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_25);  sub_tensor = rsqrt_25 = None
        convert_element_type_default_2: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        add_tensor_1: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_194, convert_element_type_default_2);  add_194 = convert_element_type_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_int: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 6);  view_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_1: "f32[]" = torch.ops.aten.select.int(select_int, 0, 0);  select_int = None
        mul_tensor_4: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_tensor_1, select_int_1);  add_tensor_1 = select_int_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        select_int_2: "f32[]" = torch.ops.aten.select.int(slice_tensor, 0, 0);  slice_tensor = None
        mul_tensor_5: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, select_int_2);  select_int_2 = None
        add_tensor_2: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        reshape_default_2: "bf16[6144, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        return reshape_default_2



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
