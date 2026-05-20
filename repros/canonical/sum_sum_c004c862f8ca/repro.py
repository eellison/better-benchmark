"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train
Pattern hash: c004c862f8ca
Shape hash: 09e879e9
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 4096], f32), T([8, 512, 4096], f32), T([4096], f32), T([8, 512, 4096], f32), T([8, 512, 1], f32), S([8, 512, 4096]), S([4096, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, mm_138: "f32[4096, 4096]", mul_437: "f32[8, 512, 4096]", primals_19: "f32[4096]", mul_4: "f32[8, 512, 4096]", div_38: "f32[8, 512, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        reshape_default: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_138, _shape_param_0);  mm_138 = _shape_param_0 = None
        add_tensor: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_437, reshape_default);  mul_437 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor, primals_19);  add_tensor = primals_19 = None
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, 4096)
        sum_dim_int_list: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_4, sum_dim_int_list_1);  mul_4 = sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_38, sub_tensor_1);  div_38 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        reshape_default_1: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None
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
