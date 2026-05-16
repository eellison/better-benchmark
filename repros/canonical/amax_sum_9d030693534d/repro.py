"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=amax, ranges=['8', '12', '64', '64', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['8', '12', '64', '64', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm: "f32[6144, 64, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1271 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        reshape_default: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.reshape.default(bmm, [8, 12, 64, 64, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1293 in forward, code: logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        amax_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.amax.default(reshape_default, [-1], True)
        abs_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax_default)
        eq_scalar: "b8[8, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_default, inf);  abs_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq_scalar, full_default, amax_default);  eq_scalar = full_default = amax_default = None
        sub_tensor: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(reshape_default, where_self)
        exp_default: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True);  exp_default = None
        log_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        add_tensor: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log_default, where_self);  log_default = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1294 in forward, code: attention_probs = torch.exp(query_key_dots - logits)
        sub_tensor_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(reshape_default, add_tensor);  reshape_default = add_tensor = None
        exp_default_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1307 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_default: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_default_1, [8, 12, 64, 64, 128]);  exp_default_1 = None
        reshape_default_1: "f32[6144, 64, 128]" = torch.ops.aten.reshape.default(expand_default, [6144, 64, 128]);  expand_default = None
        return reshape_default_1


def _default_make_inputs():
    return [
    torch.randn([6144, 64, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
