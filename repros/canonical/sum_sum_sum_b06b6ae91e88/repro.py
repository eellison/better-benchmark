"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train
Pattern hash: b06b6ae91e88
Shape hash: 309fcaca
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 1024], f32), T([8192, 4096], f32), T([64, 128, 1024], f32), T([64, 128, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), T([1024], f32), T([64, 128, 1024], f32), T([64, 128, 1], f32), T([64, 128, 1], f32), T([64, 128, 1024], f32), S([1024]), S([4096]), S([1024]), S([1024]), S([64, 128, 1024]), S([1024]), S([64, 128, 1024]), S([1024]), S([64, 128, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, view_22: "f32[8192, 1024]", view_25: "f32[8192, 4096]", view_27: "f32[64, 128, 1024]", mul_8: "f32[64, 128, 1024]", view_28: "f32[8192, 1024]", view_40: "f32[8192, 1024]", mm_6: "f32[8192, 1024]", view_43: "f32[8192, 1024]", mm_8: "f32[8192, 1024]", view_47: "f32[8192, 1024]", mm_10: "f32[8192, 1024]", primals_1: "f32[1024]", primals_3: "f32[64, 128, 1024]", getitem_1: "f32[64, 128, 1]", rsqrt: "f32[64, 128, 1]", add_7: "f32[64, 128, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_default: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_22, [1, 0])
        sum_dim_int_list: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
        reshape_default: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        permute_default_1: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_25, [1, 0])
        sum_dim_int_list_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_25, [0], True);  view_25 = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:372 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_tensor: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(view_27, mul_8);  mul_8 = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_27, [0, 1]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_2: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_28, [1, 0])
        sum_dim_int_list_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
        reshape_default_2: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        permute_default_3: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_40, [1, 0])
        sum_dim_int_list_5: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
        reshape_default_3: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_3);  sum_dim_int_list_5 = _shape_param_3 = None
        reshape_default_4: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_6, _shape_param_4);  mm_6 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        permute_default_4: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_43, [1, 0])
        sum_dim_int_list_6: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_43, [0], True);  view_43 = None
        reshape_default_5: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_5);  sum_dim_int_list_6 = _shape_param_5 = None
        reshape_default_6: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_8, _shape_param_6);  mm_8 = _shape_param_6 = None
        add_tensor: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(reshape_default_4, reshape_default_6);  reshape_default_4 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_5: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_47, [1, 0])
        sum_dim_int_list_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_47, [0], True);  view_47 = None
        reshape_default_7: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_7);  sum_dim_int_list_7 = _shape_param_7 = None
        reshape_default_8: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_10, _shape_param_8);  mm_10 = _shape_param_8 = None
        add_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_8);  add_tensor = reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_1);  primals_1 = None
        mul_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 1024)
        sum_dim_int_list_8: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True)
        sub_tensor: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul_tensor_3: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_4: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_3);  mul_tensor_1 = None
        sum_dim_int_list_9: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, sum_dim_int_list_9);  sum_dim_int_list_9 = None
        sub_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list_8);  mul_tensor_2 = sum_dim_int_list_8 = None
        sub_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_5);  sub_tensor_1 = mul_tensor_5 = None
        div_tensor: "f32[64, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_tensor_6: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_7: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_3);  mul_tensor_3 = None
        sum_dim_int_list_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1]);  add_tensor_1 = None
        add_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_7, mul_tensor_6);  add_7 = mul_tensor_6 = None
        return (permute_default, reshape_default, permute_default_1, reshape_default_1, sum_dim_int_list_2, sum_dim_int_list_3, permute_default_2, reshape_default_2, permute_default_3, reshape_default_3, permute_default_4, reshape_default_5, permute_default_5, reshape_default_7, sum_dim_int_list_10, sum_dim_int_list_11, add_tensor_2)


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
