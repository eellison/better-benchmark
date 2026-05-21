"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 22173e4afc47
Shape hash: fa12b77a
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
    def forward(self, bmm_69: "f32[1536, 128, 128]", gt_2: "b8[128, 12, 128, 128]", bmm: "f32[1536, 128, 128]", eq: "b8[128, 1, 128, 128]", amax: "f32[128, 12, 128, 1]", sum_1: "f32[128, 12, 128, 1]", full_default_13: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        reshape_default: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_69, _shape_param_0);  bmm_69 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_default: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        reshape_default_1: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        div_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(reshape_default_1, 8.0);  reshape_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        full_default: "f32[]" = torch.ops.aten.full.default([], -1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_tensor);  full_default = div_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        sub_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_self, amax);  where_self = amax = None
        exp_default: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor_1: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_1);  exp_default = sum_1 = None
        mul_tensor_2: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_tensor_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_tensor_1);  div_tensor_1 = None
        fma_default: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_self_1: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default_13, fma_default);  eq = full_default_13 = fma_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_tensor_2: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_self_1, 8.0);  where_self_1 = None
        reshape_default_2: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_tensor_2, _shape_param_2);  div_tensor_2 = _shape_param_2 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([1536, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [128, 12, 128, 128], dtype=torch.bool, device='cuda'),
    torch.randn([1536, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [128, 1, 128, 128], dtype=torch.bool, device='cuda'),
    torch.randn([128, 12, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 12, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    [128, 12, 128, 128],  # _shape_param_0
    [128, 12, 128, 128],  # _shape_param_1
    [1536, 128, 128],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
