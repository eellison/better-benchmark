"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: ac8426c3d224
Shape hash: b6788913
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
_shapes_config = "(T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([768], f32), T([128, 128, 768], f32), T([128, 128, 1], f32), T([128, 128, 768], f32), T([], f32), T([128, 128, 768], b8), T([128, 128], i64, gen=Index(3)), T([128, 128], i64, gen=Index(20005)), S([128, 128, 768]), S([128, 128, 768]), S([128, 128, 768]), S([768]), S([768]), S([128, 128, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_142: "f32[16384, 768]", mm_144: "f32[16384, 768]", mm_146: "f32[16384, 768]", primals_6: "f32[768]", sub: "f32[128, 128, 768]", sqrt: "f32[128, 128, 1]", add_201: "f32[128, 128, 768]", full_default_13: "f32[]", gt_1: "b8[128, 128, 768]", primals_5: "i64[128, 128]", primals_1: "i64[128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        reshape_default: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_142, _shape_param_0);  mm_142 = _shape_param_0 = None
        reshape_default_1: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_144, _shape_param_1);  mm_144 = _shape_param_1 = None
        add_tensor: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        reshape_default_2: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_146, _shape_param_2);  mm_146 = _shape_param_2 = None
        add_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1], True)
        reshape_default_3: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None
        mul_tensor: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_6, sub)
        add_tensor_2: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt, 1e-06)
        div_tensor: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor, add_tensor_2);  mul_tensor = None
        div_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_tensor, add_tensor_2);  div_tensor = None
        neg_default: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_tensor_1)
        mul_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None
        sum_dim_int_list_1: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        mul_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_tensor_2, primals_6);  primals_6 = None
        mul_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_tensor_2, sub);  div_tensor_2 = None
        sum_dim_int_list_2: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_4);  sum_dim_int_list_2 = _shape_param_4 = None
        neg_default_1: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_tensor_2)
        sum_dim_int_list_3: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_default_1, [2], True);  neg_default_1 = None
        add_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_201, mul_tensor_2);  add_201 = mul_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_scalar: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt, 2)
        div_tensor_3: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_dim_int_list_1, mul_scalar);  sum_dim_int_list_1 = mul_scalar = None
        eq_scalar: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt, 0);  sqrt = None
        where_self: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_scalar, full_default_13, div_tensor_3);  eq_scalar = div_tensor_3 = None
        mul_scalar_1: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_self, 0.002607561929595828);  where_self = None
        mul_tensor_4: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_scalar_1, sub);  mul_scalar_1 = sub = None
        add_tensor_4: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_4);  add_tensor_3 = mul_tensor_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_default: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_dim_int_list_3, _shape_param_5);  sum_dim_int_list_3 = _shape_param_5 = None
        div_scalar: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        add_tensor_5: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_4, div_scalar);  add_tensor_4 = div_scalar = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:33 in forward, code: return self.dropout(x)
        convert_element_type_default: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor_5: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_5, mul_tensor_5);  add_tensor_5 = mul_tensor_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32 in forward, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        eq_scalar_1: "b8[128, 128]" = torch.ops.aten.eq.Scalar(primals_5, 0)
        unsqueeze_default: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[128, 128, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_13, mul_tensor_6);  unsqueeze_default = None
        full_default: "f32[3, 768]" = torch.ops.aten.full.default([3, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[3, 768]" = torch.ops.aten.index_put.default(full_default, [primals_5], where_self_1, True);  full_default = primals_5 = where_self_1 = None
        eq_scalar_2: "b8[128, 128]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default_1: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[128, 128, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_13, mul_tensor_6);  unsqueeze_default_1 = full_default_13 = mul_tensor_6 = None
        full_default_14: "f32[20005, 768]" = torch.ops.aten.full.default([20005, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[20005, 768]" = torch.ops.aten.index_put.default(full_default_14, [primals_1], where_self_2, True);  full_default_14 = primals_1 = where_self_2 = None
        return (reshape_default_3, reshape_default_4, index_put_default, index_put_default_1)



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
