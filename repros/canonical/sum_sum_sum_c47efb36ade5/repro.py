"""
Standalone repro captured via capture_hook.
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([16384, 2048], bf16), T([16384, 2048], bf16), T([16384, 768], bf16, stride=(1536, 1)), T([16384, 768], bf16), T([16384, 768], bf16, stride=(1536, 1)), T([128, 1536, 2048], bf16), T([16384], i64, max=100), T([2048, 8], f32), T([2048, 1], f32), T([2048, 8], i64, max=100), T([2048, 128], bf16), T([2048, 1], f32), T([2048, 1], f32), T([128, 2048], bf16))"

class Repro(torch.nn.Module):
    def forward(self, where_16: "bf16[16384, 2048]", _grouped_mm_7: "bf16[16384, 2048]", getitem_58: "bf16[16384, 768]", _grouped_mm_9: "bf16[16384, 768]", getitem_59: "bf16[16384, 768]", primals_46: "bf16[128, 1536, 2048]", getitem_57: "i64[16384]", div_13: "f32[2048, 8]", sum_11: "f32[2048, 1]", getitem_55: "i64[2048, 8]", mm_19: "bf16[2048, 128]", amax_3: "f32[2048, 1]", sum_10: "f32[2048, 1]", primals_45: "bf16[128, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:458 in grouped_mm_experts_forward, code: weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
        mul_tensor: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_16, _grouped_mm_7);  where_16 = _grouped_mm_7 = None
        sum_dim_int_list: "bf16[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True);  mul_tensor = None
        squeeze_dim: "bf16[16384]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, -1);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_default: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(getitem_58, torch.float32);  getitem_58 = None
        neg_default: "f32[16384, 768]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[16384, 768]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[16384, 768]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[16384, 768]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor)
        convert_element_type_default_1: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:515 in _default_apply_gate, code: return self.act_fn(gate) * up  # (S, intermediate_dim)
        mul_tensor_1: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(_grouped_mm_9, convert_element_type_default_1);  convert_element_type_default_1 = None
        mul_tensor_2: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(_grouped_mm_9, getitem_59);  _grouped_mm_9 = getitem_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_default_2: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float32);  mul_tensor_2 = None
        reciprocal_default: "f32[16384, 768]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor_3: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_4: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_3);  convert_element_type_default_2 = None
        sub_tensor: "f32[16384, 768]" = torch.ops.aten.sub.Tensor(1, mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_5: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, sub_tensor);  convert_element_type_default = sub_tensor = None
        add_tensor_1: "f32[16384, 768]" = torch.ops.aten.add.Tensor(mul_tensor_5, 1);  mul_tensor_5 = None
        mul_tensor_6: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_1);  mul_tensor_4 = add_tensor_1 = None
        convert_element_type_default_3: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:514 in _default_apply_gate, code: gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
        cat_default: "bf16[16384, 1536]" = torch.ops.aten.cat.default([convert_element_type_default_3, mul_tensor_1], 1);  convert_element_type_default_3 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_default: "bf16[128, 2048, 1536]" = torch.ops.aten.permute.default(primals_46, [0, 2, 1]);  primals_46 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:6781 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_default_1: "bf16[128, 1536, 2048]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:391 in grouped_mm_experts_forward, code: sample_weights_g = sample_weights[perm]
        full_default: "bf16[16384]" = torch.ops.aten.full.default([16384], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "bf16[16384]" = torch.ops.aten.index_put.default(full_default, [getitem_57], squeeze_dim, True);  full_default = getitem_57 = squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:385 in grouped_mm_experts_forward, code: sample_weights = top_k_weights.reshape(-1)  # (S,)
        reshape_default: "bf16[2048, 8]" = torch.ops.aten.reshape.default(index_put_default, [2048, 8]);  index_put_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:270 in forward, code: router_top_value = router_top_value.to(router_logits.dtype)
        convert_element_type_default_4: "f32[2048, 8]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:269 in forward, code: router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        div_tensor_1: "f32[2048, 8]" = torch.ops.aten.div.Tensor(div_13, sum_11);  div_13 = None
        neg_default_1: "f32[2048, 8]" = torch.ops.aten.neg.default(convert_element_type_default_4)
        mul_tensor_7: "f32[2048, 8]" = torch.ops.aten.mul.Tensor(neg_default_1, div_tensor_1);  neg_default_1 = div_tensor_1 = None
        div_tensor_2: "f32[2048, 8]" = torch.ops.aten.div.Tensor(convert_element_type_default_4, sum_11);  convert_element_type_default_4 = sum_11 = None
        sum_dim_int_list_1: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [1], True);  mul_tensor_7 = None
        expand_default: "f32[2048, 8]" = torch.ops.aten.expand.default(sum_dim_int_list_1, [2048, 8]);  sum_dim_int_list_1 = None
        add_tensor_2: "f32[2048, 8]" = torch.ops.aten.add.Tensor(div_tensor_2, expand_default);  div_tensor_2 = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:267 in forward, code: router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        full_default_1: "f32[2048, 128]" = torch.ops.aten.full.default([2048, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_src: "f32[2048, 128]" = torch.ops.aten.scatter.src(full_default_1, -1, getitem_55, add_tensor_2);  full_default_1 = getitem_55 = add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:266 in forward, code: router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        convert_element_type_default_5: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        sub_tensor_1: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_5, amax_3);  convert_element_type_default_5 = amax_3 = None
        exp_default_1: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        div_tensor_3: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp_default_1, sum_10);  exp_default_1 = sum_10 = None
        mul_tensor_8: "f32[2048, 128]" = torch.ops.aten.mul.Tensor(scatter_src, div_tensor_3);  scatter_src = None
        sum_dim_int_list_2: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [-1], True)
        neg_default_2: "f32[2048, 128]" = torch.ops.aten.neg.default(div_tensor_3);  div_tensor_3 = None
        fma_default: "f32[2048, 128]" = torch.ops.prims.fma.default(neg_default_2, sum_dim_int_list_2, mul_tensor_8);  neg_default_2 = sum_dim_int_list_2 = mul_tensor_8 = None
        convert_element_type_default_6: "bf16[2048, 128]" = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_default_2: "bf16[2048, 128]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_default_3: "bf16[128, 2048]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (cat_default, permute_default_1, convert_element_type_default_6, permute_default_3)


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
