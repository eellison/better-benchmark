"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_GPT2_train
Pattern hash: bd8cc231fc9e
Shape hash: 4012d8f1
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
_shapes_config = "(T([2048, 768], f32), T([768], f32), T([4, 512, 768], f32), T([4, 512, 1], f32), T([4, 512, 768], f32), T([4, 512, 768], b8), T([1, 512], i64, gen=Index(1024)), T([4, 512], i64, gen=Index(50257)), T([50257, 768], f32), S([4, 512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_97: "f32[2048, 768]", primals_4: "f32[768]", mul_2: "f32[4, 512, 768]", div_24: "f32[4, 512, 1]", add_149: "f32[4, 512, 768]", gt: "b8[4, 512, 768]", unsqueeze: "i64[1, 512]", primals_1: "i64[4, 512]", mm_1: "f32[50257, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_97, _shape_param_0);  mm_97 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_4);  primals_4 = None
        mul_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_tensor_1);  div_24 = sub_tensor_1 = None
        mul_tensor_5: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default, mul_2);  mul_2 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None
        add_tensor: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_149, mul_tensor_4);  add_149 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        convert_element_type_default: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_6: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_7: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_6);  add_tensor = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        sum_dim_int_list_4: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(unsqueeze, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        where_self: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default_1: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_1, [unsqueeze], where_self, True);  full_default_1 = unsqueeze = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        eq_scalar_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default_1: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, mul_tensor_7);  unsqueeze_default_1 = full_default = mul_tensor_7 = None
        full_default_2: "f32[50257, 768]" = torch.ops.aten.full.default([50257, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[50257, 768]" = torch.ops.aten.index_put.default(full_default_2, [primals_1], where_self_1, True);  full_default_2 = primals_1 = where_self_1 = None
        add_tensor_1: "f32[50257, 768]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_1);  mm_1 = index_put_default_1 = None
        return (sum_dim_int_list_3, sum_dim_int_list_2, index_put_default, add_tensor_1)



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
