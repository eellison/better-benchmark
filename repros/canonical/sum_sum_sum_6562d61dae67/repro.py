"""
Standalone repro captured via capture_hook.
Label: hf_pythia_410m_train
Pattern hash: 6562d61dae67
Shape hash: 5a29e84a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_181: "f16[2048, 1024]", _shape_param_0, primals_22: "f16[1024]", add_11: "f16[4, 512, 1024]", getitem_17: "f32[4, 512, 1]", rsqrt_2: "f32[4, 512, 1]", add_485: "f16[4, 512, 1024]", mm_185: "f16[2048, 1024]", _shape_param_1, primals_16: "f16[1024]", _shape_param_2, primals_8: "f16[1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        reshape_default: "f16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_181, _shape_param_0);  mm_181 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        convert_element_type_default_1: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_22, torch.float32);  primals_22 = None
        mul_tensor: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, convert_element_type_default_1);  convert_element_type_default = convert_element_type_default_1 = None
        mul_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        convert_element_type_default_2: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32);  add_11 = None
        sub_tensor: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_17);  convert_element_type_default_2 = getitem_17 = None
        mul_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_2);  sub_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_3: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None
        mul_tensor_5: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  sub_tensor_2 = None
        convert_element_type_default_3: "f16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.float16);  mul_tensor_5 = None
        add_tensor: "f16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_485, convert_element_type_default_3);  add_485 = convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_185, _shape_param_1);  mm_185 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        convert_element_type_default_4: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        convert_element_type_default_5: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_16, torch.float32);  primals_16 = None
        mul_tensor_6: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, convert_element_type_default_5);  convert_element_type_default_4 = convert_element_type_default_5 = None
        mul_tensor_7: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 1024)
        sum_dim_int_list_2: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2], True)
        mul_tensor_8: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_6, mul_tensor_2);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [2], True);  mul_tensor_8 = None
        mul_tensor_9: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_3);  mul_tensor_2 = sum_dim_int_list_3 = None
        sub_tensor_3: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_7, sum_dim_int_list_2);  mul_tensor_7 = sum_dim_int_list_2 = None
        sub_tensor_4: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_9);  sub_tensor_3 = mul_tensor_9 = None
        mul_tensor_10: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_4);  div_tensor = sub_tensor_4 = None
        convert_element_type_default_6: "f16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.float16);  mul_tensor_10 = None
        add_tensor_1: "f16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_6);  add_tensor = convert_element_type_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        reshape_default_2: "f16[2048, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_2);  add_tensor_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        permute_default: "f16[1024, 1024]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_1: "f16[1024, 1024]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [4, 512, 1024],  # _shape_param_0
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [4, 512, 1024],  # _shape_param_1
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    [2048, 1024],  # _shape_param_2
    torch.randn([1024, 1024], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
