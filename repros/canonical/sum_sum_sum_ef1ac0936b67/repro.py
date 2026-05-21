"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_train
Pattern hash: ef1ac0936b67
Shape hash: 353519f1
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
_shapes_config = "(T([12000, 384], f32), T([12000, 384], f32), T([12000, 384], f32), T([8, 1500, 384], f32, stride=(576000, 1, 1500)), T([8, 1500, 1], f32, stride=(1504, 1, 1)), T([8, 1500, 1], f32, stride=(1504, 1, 1)), T([384], f32), T([8, 1500, 384], f32, stride=(576000, 1, 1500)), S([8, 1500, 384]), S([8, 1500, 384]), S([8, 1500, 384]))"

class Repro(torch.nn.Module):
    def forward(self, mm_7: "f32[12000, 384]", mm_10: "f32[12000, 384]", mm_11: "f32[12000, 384]", primals_3: "f32[8, 1500, 384]", getitem_1: "f32[8, 1500, 1]", rsqrt: "f32[8, 1500, 1]", primals_1: "f32[384]", add_9: "f32[8, 1500, 384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        reshape_default: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_7, _shape_param_0);  mm_7 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        reshape_default_1: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_10, _shape_param_1);  mm_10 = _shape_param_1 = None
        add_tensor: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        reshape_default_2: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_11, _shape_param_2);  mm_11 = _shape_param_2 = None
        add_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_tensor: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul_tensor: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_1);  primals_1 = None
        mul_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 384)
        sum_dim_int_list: "f32[8, 1500, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True)
        mul_tensor_3: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor);  mul_tensor_1 = None
        sum_dim_int_list_1: "f32[8, 1500, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list);  mul_tensor_2 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[8, 1500, 1]" = torch.ops.aten.div.Tensor(rsqrt, 384);  rsqrt = None
        mul_tensor_5: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor);  mul_tensor = None
        sum_dim_int_list_2: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1]);  add_tensor_1 = None
        add_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_9, mul_tensor_5);  add_9 = mul_tensor_5 = None
        return (sum_dim_int_list_3, sum_dim_int_list_2, add_tensor_2)



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
