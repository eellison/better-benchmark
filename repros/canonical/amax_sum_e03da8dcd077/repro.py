"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer
Pattern hash: e03da8dcd077
Shape hash: f0463697
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
_shapes_config = "(T([768, 64, 128], f16), T([4096, 768], f16), S([1, 12, 64, 64, 128]), S([1, 12, 64, 64, 128]), S([768, 64, 128]), S([1, 4096, 768]), S([1, 4096, 12, 64]), S([1, 12, 64, 64, 64]), S([1, 12, 64, 128, 64]), S([768, 128, 64]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_10: "f16[768, 64, 128]", mm_16: "f16[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1236 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        reshape_default: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.reshape.default(bmm_10, _shape_param_0);  bmm_10 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1258 in forward, code: logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        convert_element_type_default: "f32[1, 12, 64, 64, 128]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32)
        amax_default: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        abs_default: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax_default)
        eq_scalar: "b8[1, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_default, inf);  abs_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq_scalar, full_default, amax_default);  eq_scalar = full_default = amax_default = None
        sub_tensor: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default, where_self);  convert_element_type_default = None
        exp_default: "f32[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True);  exp_default = None
        log_default: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        add_tensor: "f32[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log_default, where_self);  log_default = where_self = None
        convert_element_type_default_1: "f16[1, 12, 64, 64, 1]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float16);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1259 in forward, code: attention_probs = torch.exp(query_key_dots - logits)
        sub_tensor_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(reshape_default, convert_element_type_default_1);  reshape_default = convert_element_type_default_1 = None
        exp_default_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1268 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_default: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_default_1, _shape_param_1);  exp_default_1 = _shape_param_1 = None
        reshape_default_1: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1164 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default_2: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_16, _shape_param_3);  mm_16 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        reshape_default_3: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_4);  reshape_default_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        reshape_default_4: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_tensor: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_4, 2, -1, 9223372036854775807)
        slice_tensor_1: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_4, 2, 0, -1)
        cat_default: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_default_1: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_default, reshape_default_4], 3);  cat_default = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1268 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_default_1: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_default_1, _shape_param_6);  cat_default_1 = _shape_param_6 = None
        reshape_default_5: "f16[768, 128, 64]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_7);  expand_default_1 = _shape_param_7 = None
        return (reshape_default_1, reshape_default_5)



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
