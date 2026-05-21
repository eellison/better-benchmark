"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 892d3e9ca1c8
Shape hash: 0a05ef0e
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
_shapes_config = "(T([1536, 128, 128], f32), T([128, 1, 128, 128], b8), T([], f32), T([61], i64), S([128, 12, 128, 128]), S([128, 12, 128, 128]), S([1536, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_22: "f32[1536, 128, 128]", eq: "b8[128, 1, 128, 128]", full_default: "f32[]", inductor_seeds_default: "i64[61]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        reshape_default: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, _shape_param_0);  bmm_22 = _shape_param_0 = None
        div_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(reshape_default, 8.0);  reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_self: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_tensor);  eq = full_default = div_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax_default: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_self, [-1], True)
        sub_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_self, amax_default);  where_self = amax_default = None
        exp_default: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 56);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor_1);  gt_scalar = div_tensor_1 = None
        mul_tensor_1: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_default: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        reshape_default_1: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        return reshape_default_1



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
