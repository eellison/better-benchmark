"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=amax, ranges=['4', '64', '512', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['4', '64', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, cat_2: "f32[4, 64, 512, 513]", getitem_4: "f32[4, 64, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:264 in eager_attention_forward, code: combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        sub_tensor: "f32[4, 64, 512, 513]" = torch.ops.aten.sub.Tensor(cat_2, getitem_4);  cat_2 = getitem_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:265 in eager_attention_forward, code: probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        amax_default: "f32[4, 64, 512, 1]" = torch.ops.aten.amax.default(sub_tensor, [-1], True)
        sub_tensor_1: "f32[4, 64, 512, 513]" = torch.ops.aten.sub.Tensor(sub_tensor, amax_default);  sub_tensor = amax_default = None
        exp_default: "f32[4, 64, 512, 513]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[4, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[4, 64, 512, 513]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:266 in eager_attention_forward, code: scores = probs[..., :-1]  # we drop the sink here
        slice_tensor: "f32[4, 64, 512, 512]" = torch.ops.aten.slice.Tensor(div_tensor, 3, 0, -1);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:267 in eager_attention_forward, code: attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
        clone_default: "f32[4, 64, 512, 512]" = torch.ops.aten.clone.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:268 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(clone_default, [4, 64, 512, 512]);  clone_default = None
        reshape_default: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_default, [256, 512, 512]);  expand_default = None
        return reshape_default


def _default_make_inputs():
    return [
    torch.randn([4, 64, 512, 513], dtype=torch.float32, device='cuda'),
    torch.randn([4, 64, 512, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
