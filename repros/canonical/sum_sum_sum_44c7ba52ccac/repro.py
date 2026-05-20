"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train
Pattern hash: 44c7ba52ccac
Shape hash: 91a24a50
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8, 1024, 1024], f32), T([8, 1024, 1024], f32), T([8192, 1024], f32), T([8192, 4096], f32), T([8, 1024, 1024], f32), T([8, 1024, 1024], f32), T([8192, 1024], f32), T([8, 16, 1024, 64], f32, stride=(1048576, 64, 1024, 1)), T([8192, 1024], f32), T([8, 1024, 1024], f32), T([8192, 1024], f32), T([8, 1024, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), S([1024]), S([4096]), S([8, 1024, -1]), S([8192, 1024]), S([1024]), S([8192, 1024]), S([1024]), S([8, 1024, 1024]), S([1024]), S([8, 1024, 1024]), S([1024]), S([8, 1024, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[8, 1024, 1024]", mul_9: "f32[8, 1024, 1024]", view_16: "f32[8192, 1024]", view_19: "f32[8192, 4096]", add_9: "f32[8, 1024, 1024]", mul_2: "f32[8, 1024, 1024]", view_22: "f32[8192, 1024]", getitem: "f32[8, 16, 1024, 64]", view_28: "f32[8192, 1024]", primals_1: "f32[8, 1024, 1024]", mm_6: "f32[8192, 1024]", mul_32: "f32[8, 1024, 1024]", view_31: "f32[8192, 1024]", mm_8: "f32[8192, 1024]", view_35: "f32[8192, 1024]", mm_10: "f32[8192, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, mul_9);  mul_9 = None
        sum_dim_int_list: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[1024]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:385 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_default: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_16, [1, 0])
        sum_dim_int_list_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_16, [0], True);  view_16 = None
        reshape_default: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_0);  sum_dim_int_list_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:383 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        permute_default_1: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_19, [1, 0])
        sum_dim_int_list_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_19, [0], True);  view_19 = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:364 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(add_9, mul_2);  mul_2 = None
        sum_dim_int_list_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_9, [0, 1]);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_2: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_22, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_3: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_2: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_2);  permute_default_3 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_3: "f32[8192, 1024]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        sum_dim_int_list_6: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
        reshape_default_4: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_4);  sum_dim_int_list_6 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        permute_default_4: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_28, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_5: "f32[8192, 1024]" = torch.ops.aten.reshape.default(primals_1, _shape_param_5);  primals_1 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        sum_dim_int_list_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
        reshape_default_6: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_6);  sum_dim_int_list_7 = _shape_param_6 = None
        reshape_default_7: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(mm_6, _shape_param_7);  mm_6 = _shape_param_7 = None
        add_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_32, reshape_default_7);  mul_32 = reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        permute_default_5: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_31, [1, 0])
        sum_dim_int_list_8: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_31, [0], True);  view_31 = None
        reshape_default_8: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_8);  sum_dim_int_list_8 = _shape_param_8 = None
        reshape_default_9: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(mm_8, _shape_param_9);  mm_8 = _shape_param_9 = None
        add_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_9);  add_tensor = reshape_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_6: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_35, [1, 0])
        sum_dim_int_list_9: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_35, [0], True);  view_35 = None
        reshape_default_10: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_10);  sum_dim_int_list_9 = _shape_param_10 = None
        reshape_default_11: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(mm_10, _shape_param_11);  mm_10 = _shape_param_11 = None
        add_tensor_2: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_11);  add_tensor_1 = reshape_default_11 = None
        return (sum_dim_int_list, sum_dim_int_list_1, permute_default, reshape_default, permute_default_1, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_2, reshape_default_3, reshape_default_4, permute_default_4, reshape_default_5, reshape_default_6, permute_default_5, reshape_default_8, permute_default_6, reshape_default_10, add_tensor_2)


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
