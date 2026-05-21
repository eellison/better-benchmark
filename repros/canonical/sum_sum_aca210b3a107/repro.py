"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_train
Pattern hash: aca210b3a107
Shape hash: dde5ac79
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
_shapes_config = "(T([12000, 384], f32), T([12000, 384], f32), T([8, 1500, 384], f32, stride=(576000, 1, 1500)), T([8, 1500, 1], f32, stride=(1504, 1, 1)), T([8, 1500, 1], f32, stride=(1504, 1, 1)), T([384], f32), T([8, 1500, 384], f32, stride=(576000, 1, 1500)), S([8, 1500, 384]), S([8, 1500, 384]), S([12000, 384]))"

class Repro(torch.nn.Module):
    def forward(self, mm_3: "f32[12000, 384]", addmm_2: "f32[12000, 384]", primals_3: "f32[8, 1500, 384]", getitem_7: "f32[8, 1500, 1]", rsqrt_1: "f32[8, 1500, 1]", primals_11: "f32[384]", tangents_1: "f32[8, 1500, 384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_3, _shape_param_0);  mm_3 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_1: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_1);  addmm_2 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(primals_3, reshape_default_1);  primals_3 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_tensor: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_7);  add_tensor = getitem_7 = None
        mul_tensor: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_1);  sub_tensor = None
        mul_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(reshape_default, primals_11);  reshape_default = primals_11 = None
        mul_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 384)
        sum_dim_int_list: "f32[8, 1500, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True)
        mul_tensor_3: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor);  mul_tensor_1 = None
        sum_dim_int_list_1: "f32[8, 1500, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, sum_dim_int_list_1);  mul_tensor = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list);  mul_tensor_2 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[8, 1500, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 384);  rsqrt_1 = None
        mul_tensor_5: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(tangents_1, mul_tensor_5);  tangents_1 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        clone_default: "f32[8, 1500, 384]" = torch.ops.aten.clone.default(add_tensor_1, memory_format = torch.contiguous_format);  add_tensor_1 = None
        reshape_default_2: "f32[12000, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
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
