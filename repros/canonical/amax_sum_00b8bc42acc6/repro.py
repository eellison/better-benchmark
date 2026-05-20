"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train
Pattern hash: 00b8bc42acc6
Shape hash: 5da5141f
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 1536], f32), T([192, 512, 512], f32), T([73], i64, max=73), S([8, 512, 1536]), S([8, 512, 24, -1]), S([-1, 512, 64]), S([-1, 24, 512, 512]), S([-1, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f32[4096, 1536]", bmm: "f32[192, 512, 512]", inductor_seeds_default: "i64[73]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        reshape_default: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_0);  addmm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        reshape_default_1: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_default: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        clone_default: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        reshape_default_3: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm, _shape_param_3);  bmm = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default: "b8[8, 1, 512, 512]" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default, full_default_1, reshape_default_3);  full_default = full_default_1 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_default: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_self, [-1], True)
        sub_tensor: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_self, amax_default);  where_self = amax_default = None
        exp_default: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor);  gt_scalar = div_tensor = None
        mul_tensor_1: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        reshape_default_4: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_4);  mul_tensor_1 = _shape_param_4 = None
        return (reshape_default_2, reshape_default_4)


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
