"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: d5d45913d89a
Shape hash: 06b3bc3f
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_109: "f32[192, 128, 128]", gt_36: "b8[32, 6, 128, 128]", div_12: "f32[32, 6, 128, 128]", bmm_141: "f32[192, 128, 128]", gt_2: "b8[32, 6, 128, 128]", ge: "b8[1, 1, 128, 1]", full_default: "f32[]", bmm: "f32[192, 128, 128]", embedding_1: "f32[128, 128, 6]", amax: "f32[32, 6, 128, 1]", sum_1: "f32[32, 6, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_109, _shape_param_0);  bmm_109 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_12);  mul_tensor_1 = None
        sum_dim_int_list: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_12);  div_12 = None
        fma_default: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        reshape_default_1: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_default, _shape_param_1);  fma_default = _shape_param_1 = None
        reshape_default_2: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        reshape_default_3: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_4: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_141, _shape_param_4);  bmm_141 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_default_1: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_tensor_3: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_4: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default_4, mul_tensor_3);  reshape_default_4 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_default: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(ge, _shape_param_5);  ge = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default, full_default_1);  expand_default = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_5: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm, _shape_param_6);  bmm = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_default: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_default: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_default, 0);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        add_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_default, where_self);  unsqueeze_default = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default_5, add_tensor);  reshape_default_5 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        sub_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_1, amax);  add_tensor_1 = amax = None
        exp_default: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_1);  exp_default = sum_1 = None
        mul_tensor_5: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_4, div_tensor);  mul_tensor_4 = None
        sum_dim_int_list_1: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [-1], True)
        neg_default_1: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default_1: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_default_1, sum_dim_int_list_1, mul_tensor_5);  neg_default_1 = sum_dim_int_list_1 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        reshape_default_6: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_default_1, _shape_param_7);  fma_default_1 = _shape_param_7 = None
        reshape_default_7: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(reshape_default_6, _shape_param_8);  reshape_default_6 = _shape_param_8 = None
        reshape_default_8: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_9);  reshape_default_7 = _shape_param_9 = None
        return (reshape_default_3, reshape_default_8)


def _default_make_inputs():
    return [
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 6, 128, 128], dtype=torch.bool, device='cuda'),
    torch.randn([32, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 6, 128, 128], dtype=torch.bool, device='cuda'),
    torch.randint(0, 2, [1, 1, 128, 1], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 6], dtype=torch.float32, device='cuda'),
    torch.randn([32, 6, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 6, 128, 1], dtype=torch.float32, device='cuda'),
    [32, 6, 128, 128],  # _shape_param_0
    [192, 128, 128],  # _shape_param_1
    [32, 6, 128, 128],  # _shape_param_2
    [192, 128, 128],  # _shape_param_3
    [32, 6, 128, 128],  # _shape_param_4
    [32, -1, 128, 128],  # _shape_param_5
    [32, 6, 128, 128],  # _shape_param_6
    [192, 128, 128],  # _shape_param_7
    [32, 6, 128, 128],  # _shape_param_8
    [192, 128, 128],  # _shape_param_9
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
