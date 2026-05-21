"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer
Pattern hash: 9a4215b2df07
Shape hash: 31c62b63
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
_shapes_config = "(T([1, 12, 4096], i64, gen=Index(4096)), T([768, 64, 128], f16), T([], f16), T([4096, 768], f16), S([1, 12, -1, 64]), S([1, 12, 64, 64, 128]), S([1, 12, 64, 64, 128]), S([768, 64, 128]), S([1, 4096, 768]), S([1, 4096, 12, 64]), S([-1, -1, -1, 64]), S([1, 12, -1, 64, 64]), S([1, 12, 64, 128, 64]), S([768, 128, 64]))"

class Repro(torch.nn.Module):
    def forward(self, remainder_5: "i64[1, 12, 4096]", bmm_13: "f16[768, 64, 128]", arg67_1: "f16[]", mm_19: "f16[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:400 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape)
        reshape_default: "i64[1, 12, 64, 64]" = torch.ops.aten.reshape.default(remainder_5, _shape_param_0);  _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:875 in _attend, code: self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
        unsqueeze_default: "i64[1, 12, 64, 64, 1]" = torch.ops.aten.unsqueeze.default(reshape_default, -1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_tensor: "i64[1, 12, 1, 64]" = torch.ops.aten.slice.Tensor(reshape_default, 2, -1, 9223372036854775807)
        slice_tensor_1: "i64[1, 12, 63, 64]" = torch.ops.aten.slice.Tensor(reshape_default, 2, 0, -1)
        cat_default: "i64[1, 12, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_default_1: "i64[1, 12, 64, 128]" = torch.ops.aten.cat.default([cat_default, reshape_default], 3);  cat_default = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:875 in _attend, code: self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
        unsqueeze_default_1: "i64[1, 12, 64, 1, 128]" = torch.ops.aten.unsqueeze.default(cat_default_1, -2);  cat_default_1 = None
        ne_tensor: "b8[1, 12, 64, 64, 128]" = torch.ops.aten.ne.Tensor(unsqueeze_default, unsqueeze_default_1);  unsqueeze_default = unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:818 in _attend, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        reshape_default_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.reshape.default(bmm_13, _shape_param_1);  bmm_13 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:880 in _attend, code: query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)
        where_self: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.where.self(ne_tensor, reshape_default_1, arg67_1);  ne_tensor = reshape_default_1 = arg67_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:885 in _attend, code: logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        convert_element_type_default: "f32[1, 12, 64, 64, 128]" = torch.ops.prims.convert_element_type.default(where_self, torch.float32)
        amax_default: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        abs_default: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax_default)
        eq_scalar: "b8[1, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_default, inf);  abs_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq_scalar, full_default, amax_default);  eq_scalar = full_default = amax_default = None
        sub_tensor: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default, where_self_1);  convert_element_type_default = None
        exp_default: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True);  exp_default = None
        log_default: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        add_tensor: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log_default, where_self_1);  log_default = where_self_1 = None
        convert_element_type_default_1: "f16[1, 12, 64, 64, 1]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float16);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:887 in _attend, code: attention_probs = torch.exp(query_key_dots - logits)
        sub_tensor_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(where_self, convert_element_type_default_1);  where_self = convert_element_type_default_1 = None
        exp_default_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_default: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_default_1, _shape_param_2);  exp_default_1 = _shape_param_2 = None
        reshape_default_2: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_3);  expand_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:512 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default_3: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_19, _shape_param_4);  mm_19 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        reshape_default_4: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1063 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_default, [1, 1, 1, 1]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1062 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_default_2: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_5, -1);  remainder_5 = None
        expand_default_1: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default_2, _shape_param_6);  unsqueeze_default_2 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1064 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_default, 2, expand_default_1);  repeat_default = expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        reshape_default_5: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather_default, _shape_param_7);  gather_default = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_tensor_2: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_5, 2, -1, 9223372036854775807)
        slice_tensor_3: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_5, 2, 0, -1)
        cat_default_2: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_tensor_2, slice_tensor_3], 2);  slice_tensor_2 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_default_3: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_default_2, reshape_default_5], 3);  cat_default_2 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_default_2: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_default_3, _shape_param_8);  cat_default_3 = _shape_param_8 = None
        reshape_default_6: "f16[768, 128, 64]" = torch.ops.aten.reshape.default(expand_default_2, _shape_param_9);  expand_default_2 = _shape_param_9 = None
        return (reshape_default_2, reshape_default_6)



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
