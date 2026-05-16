"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=amax, ranges=['512', '128', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['512', '128', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_46: "f32[512, 128, 128]", expand_1: "f32[32, 1, 128, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:212 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        reshape_default: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, [32, 16, 128, 128]);  bmm_46 = None
        add_tensor: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default, expand_1);  reshape_default = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:214 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:213 in forward, code: attn_weights = torch.max(
        maximum_default: "f32[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_tensor, full_default);  add_tensor = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:216 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        reshape_default_1: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(maximum_default, [512, 128, 128]);  maximum_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_default: "f32[512, 128, 1]" = torch.ops.aten.amax.default(reshape_default_1, [-1], True)
        sub_tensor: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        return div_tensor


def _default_make_inputs():
    return [
    torch.randn([512, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1, 128, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
